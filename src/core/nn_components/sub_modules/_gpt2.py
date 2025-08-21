import torch
from torch import nn
from ..root_layers.layer_normalization import AddLNorm
from ..root_layers.attention_pool import MultiHeadAttention
from ..root_layers.ffn import PositionWiseFFN


class GPT2DecoderBlock(nn.Module):
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
        # # 交叉源信息注意力
        # self.xSrcAttn = MultiHeadAttention(num_heads, num_hiddens, dropout, use_bias)
        # self.addlnorm2 = AddLNorm(num_hiddens, dropout)
        # 前向
        self.PosFFN = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addlnorm3 = AddLNorm(num_hiddens, dropout)
    

    def forward(self, tgt_query, KV_Caches=None): # 函数内部对 kv_caches 作in-place操作

        if self.training:
            # train 过程中, tgt_query 是一个 shape 为 (batch_size, num_steps, d_dim) 的tensor. 时间步为 0 至 num_steps-1
            # 代表 target sequence timestep 0 至 num_steps-1

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
            assert type(KV_Caches) == dict, f'in infer mode, a dictionary as KV_Caches must be input'

            try:
                KV_Caches[self.blk_ind] = torch.cat([KV_Caches[self.blk_ind], tgt_query], dim=1) 
                # shape (1, i-1, d_dim) + (1, 1, d_dim) = (1, i, d_dim)
            except KeyError:
                KV_Caches[self.blk_ind] = tgt_query
            
            KVs = KV_Caches[self.blk_ind]

            tgt_dec = self.addlnorm1(tgt_query, self.ARselfAttn(tgt_query, KVs, KVs))
            
        tgt_dec = self.addlnorm3(tgt_dec, self.PosFFN(tgt_dec))

        return tgt_dec, KV_Caches