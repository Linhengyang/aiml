import torch
from torch import nn
from ..root_layers.add_layer_norm import AddLNorm
from ..root_layers.attention_pools import MultiHeadAttention

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
    
    inputs:
        X: (batch_size, seq_len, num_hiddens)

        valid_lens(optional): (batch_size,) since it's self-attention. 一条样本只有一个valid length, 这个 valid length 指出了
        X 在 dim 1 的 valid 长度

    returns:
        shape: (batch_size, seq_len, num_hiddens), 和输入的 X 保持一致, 因为多个Block要堆叠起来, 所以要保持shape一致
    
    explains: 
        keep batch shape at every layer's input/output through the block 
        encode source sequence time 1 to T directly to deep sequence time 1 to T, that is: 
            f(time 1 to T) --> node 1 to T on next layer
        
        自注意力Encoder Block. 对输入样本data作[自注意力-前向-norm]的深度处理, 样本从时间步1到T, 输出结果也是从时间步1到T

        单个样本内部, 时间步之间由于自注意力的机制, 作到了双向前后全连接表征.


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
        # X 作 自注意力. X shape(batch_size, seq_len, num_hiddens), 在 dim 1 的 seq_len 里, valid area 由 valid_lens(batch_size,) 确定
        # 可以先 mask, 再作 自注意力, 即对 X 在 dim 1 作 mask
        # 也可以先 自注意力, 再 mask, 即 (Q, K) --相似度计算--> S --softmax--> W, (W, V) --n_query次线性组合 on V--> output
        # 过程中, valid area 作用在 softmax 子过程里, 同样可以限制 valid area

        # 在自注意力中, 对每个token(对应每条query)而言, 都只限制了整体valid 长度.
        # 故 encoder 里, 每个token是跟序列里前/后所有valid token作相关性运算的
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
        
    inputs:
    train:
        tgt_query: (batch_size, seq_len, d_dim)
        src_enc_info(src_enc_seqs, src_valid_lens): (batch_size, seq_len, d_dim), (batch_size,)

        state:
            self-att层, 需要一些额外的信息, 满足infer过程的需要
            enc-dec-att层, 需要在这里输入encoder output相关信息, 即decoder在这里从source sequence中吸取信息
    
    returns:
        dec_O's shape: (batch_size, seq_len, d_dim), the same as dec_X
    
    explains:
        keep batch shape at every layer's input/output through the block
    '''
    def __init__(self, blk_ind, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias=False):
        super().__init__()
        # 记录当前 Block 的 id
        self.blk_ind = str(blk_ind)
        # 自回归-自注意力
        self.ARselfAttn = MultiHeadAttention(num_heads, num_hiddens, dropout, use_bias)
        self.addlnorm1 = AddLNorm(num_hiddens, dropout)
        # 交叉源信息注意力
        self.xSrcAttn = MultiHeadAttention(num_heads, num_hiddens, dropout, use_bias)
        self.addlnorm2 = AddLNorm(num_hiddens, dropout)
        # 前向
        self.PosFFN = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addlnorm3 = AddLNorm(num_hiddens, dropout)
    

    def forward(self, tgt_query, src_enc_info, KV_Caches=None): # 函数内部对 kv_caches 作in-place操作
        src_enc_seqs, src_valid_lens = src_enc_info
        # src_enc_seqs, timestep 1 - num_steps, shape as
        #   train: (batch_size, num_steps, d_dim), infer: (1, num_steps, d_dim)

        # src_valid_lens, shape as
        #   train: (batch_size,), infer: (1,)

        if self.training:
            # train 过程中, tgt_query 是一个 shape 为 (batch_size, num_steps, d_dim) 的tensor. 时间步为 0 至 num_steps-1
            # 代表 target sequence timestep 0 至 num_steps-1
            assert tgt_query.shape[-1] == src_enc_seqs.shape[-1],\
                f'training: enc output & dec input block {self.blk_ind} are not in same shape'
            
            batch_size, num_steps = tgt_query.shape[:-1]

            # decoder 的 train 过程, 在自注意力阶段需要保证 自回归: 对序列的每个token(每条query)而言, 只有在它前面的才是valid

            # train: 从0-T-1 T并行生成 1-T, 生成 step i(i=1,..,T) 的 qKV timestep分别是 q:i-1, K:0至i-1, V:0至i-1
            # 那么对于 T 条 query 的 QKV, timestep都是 0-T-1, valid length 应该分别是 1 for time0, 2 for time1, ..., T for time T-1
            # 即 [1,2,...,T]
            # 对 batch 内所有sample而言, 上述 valid length 是一样的, 所以重复 batch_size 遍, 成为一个 (batch_size, T) 的 2-D tensor
            valid_lens = torch.arange(1, num_steps+1, dtype=torch.int32, device=tgt_query.device).expand(batch_size, -1)

            # 自注意力+AddLayerNorm:
            # target info 深度表达: (batch_size, T, d_dim) --auto_regressive_self_attention + add_layernorm--> (batch_size, T, d_dim)
            tgt_dec = self.addlnorm1(tgt_query, self.ARselfAttn(tgt_query, tgt_query, tgt_query, valid_lens))
            
        else:
            # infer过程中, tgt_query 是一个 shape 为 (1, 1, d_dim) 的tensor. 对于第i(i=1,..,T)次infer, tgt_query 的时间步是 i-1
            # 代表 target sequence hat timestep i-1 
            # 以此 tgt_query 为 q, target sequence hat timestep 0 - i-1 为KV, 生成 target sequence hat timestep i

            # 对于第i次infer, i = 1, 2, ..., num_steps, 分别代表 生成 target sequence hat 的 timesteps 1, 2, ..., num_steps
            # tgt_query 的时间步是 i-1, KV 需要 target sequence hat timestep 0 - i-1, 所以需要额外输入 target sequence hat timestep 0 - i-2
            # 由于下一次 infer 需要 0-i-1, 故需要对 KV_Caches 当前 block 对应的 KV_Caches tensor 作更新, 存储给下一次infer用

            assert type(KV_Caches) == dict, f'in infer mode, a dictionary as KV_Caches must be input'

            try: # i = 2, ..., num_steps
                KV_Caches[self.blk_ind] = torch.cat([KV_Caches[self.blk_ind], tgt_query], dim=1) 
                # shape (1, i-1, d_dim) + (1, 1, d_dim) = (1, i, d_dim), 从 timesteps 看, 是 0 至 i-2 append i-1, 得到 0 至 i-1
            except KeyError: # i = 1: 从<bos> infer第一个token时, KV_Caches 于 当前 block id, 没有 target sequence hat cache
                KV_Caches[self.blk_ind] = tgt_query
            
            KVs = KV_Caches[self.blk_ind] # 用所有 已经保存的kv_caches 加上最新输入的信息(即 tgt_query) 作KV.

            # 自注意力+AddLayerNorm:
            # target info 深度表达: (1, 1, d_dim) -->self_attention--> (1, 1, d_dim)
            tgt_dec = self.addlnorm1(tgt_query, self.ARselfAttn(tgt_query, KVs, KVs))
        
        # target info 和 source info 信息交合, tgt_dec shape same with tgt_query
        tgt_dec = self.addlnorm2(tgt_dec, self.xSrcAttn(tgt_dec, src_enc_seqs, src_enc_seqs, src_valid_lens))

        return self.addlnorm3(tgt_dec, self.PosFFN(tgt_dec)), KV_Caches
        # train 模式下第二个输出是 None, eval 模式下第二个输出 是以 block ID为key, 该 block 的 所有输入tgt_query堆叠 (timestep 0 - i-1)
        # 只有该block对应的 kv 被更新