import torch
from torch import nn
from ..Base.SubModules.AddLNorm import AddLNorm
from ..Base.RootLayers.AttentionPools import MultiHeadAttention

class PositionWiseFFN(nn.Module):
    '''
    args: ffn_num_hiddens, ffn_num_outputs
        ffn_num_hiddens: the hidden size inside the MLP
        ffn_num_outputs: output size of the MLP, usually the same as the input size
    
    inputs: X

    returns: denoted as O
    
    explains:
        Perform the same MLP on every position. So only one MLP is enough.
    '''
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)
    
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))






class TransformerEncoderBlock(nn.Module):
    '''
    components: 
        1. multihead attention(self-att) 
        2. addLnorm 
        3. positionwiseFFN 
        4. addLnorm 
    args:
        num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias=False
    
    inputs: enc_X, valid_lens(optional) 
        enc_X's shape: (batch_size, seq_len, num_hiddens) 
        valid_lens(optional)'s shape: (batch_size,) since it's self-attention. 一条样本只有一个valid length
    
    returns: denoted as enc_O 
        enc_O's shape: (batch_size, seq_len, num_hiddens), the same as enc_X 
    
    explains: 
        keep batch shape at every layer's input/output through the block 
        encode source sequence time 1 to T directly to deep sequence time 1 to T, that is: 
            f(time 1 to T) --> node 1 to T on next layer
        
        自注意力Encoder Block. 对输入样本data作[自注意力-前向-norm]的深度处理, 样本从时间步1到T, 输出结果也是从时间步1到T
        单个样本内部, 时间步之间由于自注意力的机制, 作到了双向前后全连接表征.
        依靠Add+LayerNorm层, 自注意力的输入data batch shape 和输出 data batch shape 是相同的. 这样处理是方便多个Block堆叠
    '''
    def __init__(self, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias=False):
        super().__init__()
        # multi-head attention
        # input: Q(batch_size, n_query, q_size), K(batch_size, n_kv, k_size), V(batch_size, n_kv, v_size)
        # output: (batch_size, n_query, num_hiddens)
        self.attention = MultiHeadAttention(num_heads, num_hiddens, dropout, use_bias)
        # add + layer norm
        # input: (batch_size, n_query, num_hiddens)
        # output: (batch_size, n_query, num_hiddens)
        self.addlnorm1 = AddLNorm(num_hiddens, dropout)
        # PosFFN
        # input: (batch_size, n_query, num_hiddens)
        # output: (batch_size, n_query, num_hiddens)
        self.PosFFN = PositionWiseFFN(ffn_num_hiddens, num_hiddens)

        self.addlnorm2 = AddLNorm(num_hiddens, dropout)
    
    def forward(self, X, valid_lens):
        Y = self.addlnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addlnorm2(Y, self.PosFFN(Y))






class TransformerDecoderBlock(nn.Module):
    '''
    components:
        1. masked multihead attention(self-att but using a mask to be auto-regressive)
        2. addLnorm
        3. encoder-decoder attention(use sequences from decoder as queries, keys&values are from encoder)
        4. addLnorm
        5. positionwiseFFN
        6. addLnorm

    args:
        blk_ind, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias=False
        
    inputs: dec_X, state
        dec_X's shape: (batch_size, seq_len, d_dim)
        state:
            self-att层, 需要一些额外的信息, 满足infer过程的需要
            enc-dec-att层, 需要在这里输入encoder output相关信息, 即decoder在这里从source sequence中吸取信息
    
    returns: denoted as dec_O
        dec_O's shape: (batch_size, seq_len, d_dim), the same as dec_X
    
    explains:
        keep batch shape at every layer's input/output through the block
    '''
    def __init__(self, blk_ind, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias=False):
        super().__init__()
        self.blk_ind = str(blk_ind)
        self.attention1 = MultiHeadAttention(num_heads, num_hiddens, dropout, use_bias)
        self.addlnorm1 = AddLNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(num_heads, num_hiddens, dropout, use_bias)
        self.addlnorm2 = AddLNorm(num_hiddens, dropout)
        self.PosFFN = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addlnorm3 = AddLNorm(num_hiddens, dropout)
    

    def forward(self, tgt_query, src_enc_info, KV_Caches=None): # 函数内部对 kv_caches 作in-place操作
        src_enc_seqs, src_valid_lens = src_enc_info
        # src_enc_seqs timestep 1 - num_steps, shape as
        #   train: (batch_size, num_steps, d_dim), infer: (1, num_steps, d_dim)

        # src_valid_lens's shape, train: (batch_size,), infer: (1,)

        if self.training:
            # train 过程中, tgt_query 是一个 shape 为 (batch_size, num_steps, d_dim) 的tensor. 时间步为 0 至 num_steps-1
            # 代表 target sequence timestep 0 至 num_steps-1
            assert tgt_query.shape[-1] == src_enc_seqs.shape[-1],\
                f'training: enc output & dec input block {self.blk_ind} are not in same shape'
            
            batch_size, num_steps = tgt_query.shape[:-1]

            KVs, KV_Caches = tgt_query, None # 自注意力. train过程中不需要 kv_caches
            # 用 mask 限制 kv valid 部分, 实现train过程中,
            # 从0-T-1 T并行生成 1-T, 保证 step i 的 qKV timestepe分别是 q:i-1, K:0至i-1, V:0至i-1（限制了kv的valid len是i） 
            mask = torch.arange(1, num_steps+1, dtype=torch.int32, device=tgt_query.device).repeat(batch_size, 1)
        
        else:
            # infer过程中, tgt_query 是一个 shape 为 (1, 1, d_dim) 的tensor. 对于第i次infer, tgt_query 的时间步是 i-1
            # 代表 target sequence hat timestep i-1   以此 tgt_query 为 q, target sequence hat timestep 0 - i-1 为KV, 生成 target sequence hat timestep i
            assert type(KV_Caches) == dict, f'in infer mode, a dictionary as KV_Caches must be input'

            # 对于第i次infer, i = 1, 2, ..., num_steps, 分别代表 生成 target sequence hat 的 timesteps 1, 2, ..., num_steps
            # tgt_query 的时间步是 i-1, KV 需要 target sequence hat timestep 0 - i-1, 所以需要额外输入 target sequence hat timestep 0 - i-2
            # 对 KV_Caches 当前 block 对应的 KV_Caches tensor 作更新
            try:
                # i = 2, ..., num_steps
                KV_Caches[self.blk_ind] = torch.cat([KV_Caches[self.blk_ind], tgt_query], dim=1) # shape (1, i-1, d_dim) + (1, 1, d_dim) = (1, i, d_dim)
                # timesteps 是 0 至 i-2 append i-1, 得到 0 至 i-1
            except KeyError: # 从<bos> infer第一个token时, KV_Caches 于 当前 block id, 没有 target sequence hat cache
                # i = 1
                KV_Caches[self.blk_ind] = tgt_query
            
            KVs, mask = KV_Caches[self.blk_ind], None # 用所有 已经保存的kv_caches 加上最新输入的信息(即 tgt_query) 作KV. infer过程中不需要mask
        
        # target info 深度表达, Y shape same with tgt_query
        Y = self.addlnorm1(tgt_query, self.attention1(tgt_query, KVs, KVs, mask))
        
        # target info 和 source info 信息交合, Z shape same with tgt_query
        Z = self.addlnorm2(Y, self.attention2(Y, src_enc_seqs, src_enc_seqs, src_valid_lens))

        return self.addlnorm3(Z, self.PosFFN(Z)), KV_Caches # 第一个输出的 shape 和 tgt_query 相同