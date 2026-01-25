import torch
from torch import nn
from typing import Tuple
from src.core.layers.attention_pool import BidirectMHA, CausalMHA
from src.core.layers.feedforward import relu_ffn


class TransformerDecoderBlock(nn.Module):
    def __init__(self,
                 embd_size:int,
                 num_heads:int,
                 use_bias:bool,
                 max_context_size:int,
                 ffn_hidden_size:int,
                 attn_p_drop:float,
                 resid_p_drop:float,
                 use_cached_causal_mask:bool
                 ):
        super().__init__()
        self.causal_attention = CausalMHA(embd_size, num_heads, use_bias, max_context_size,
                                          attn_p_drop, resid_p_drop, False, use_cached_causal_mask)
        self.layer_norm1 = nn.LayerNorm(embd_size)
        self.cross_attention = BidirectMHA(embd_size, num_heads, attn_p_drop, use_bias)
        self.layer_norm2 = nn.LayerNorm(embd_size)
        self.relu_ffn = relu_ffn(embd_size, ffn_hidden_size, resid_p_drop)
        self.layer_norm3 = nn.LayerNorm(embd_size)

    def forward(self,
                x:torch.Tensor,
                encoded:Tuple[torch.Tensor, torch.Tensor],
                kv_cache:Tuple[torch.Tensor, torch.Tensor]|None=None,
                return_cache:bool = False,
                attention_mask:torch.Tensor|None = None):
        
        attn_result, new_kv_cache = self.causal_attention(x, kv_cache, return_cache, attention_mask)
        x_ = x + attn_result
        return


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
        self.addlnorm1 = AddLayerNorm(num_hiddens, dropout)
        # 交叉源信息注意力
        self.xSrcAttn = MultiHeadAttention(num_heads, num_hiddens, dropout, use_bias)
        self.addlnorm2 = AddLayerNorm(num_hiddens, dropout)
        # 前向
        self.PosFFN = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addlnorm3 = AddLayerNorm(num_hiddens, dropout)
    

    def forward(self, tgt_query, src_enc_info, KV_Caches=None): # 函数内部对 kv_caches 作in-place操作
        src_enc_seqs, src_valid_lens = src_enc_info
        # src_enc_seqs, timestep 1 <-> num_steps, shape as
        #   train: (batch_size, num_steps, d_dim), infer: (1, num_steps, d_dim)

        # src_valid_lens, shape as
        #   train: (batch_size,), infer: (1,)

        if self.training:
            # train 过程中, tgt_query 是一个 shape 为 (batch_size, num_steps, d_dim) 的tensor. 时间步为 0 至 num_steps-1
            # 代表 target sequence timestep 0 至 num_steps-1。不过可能是 左pad，也可能是右pad
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