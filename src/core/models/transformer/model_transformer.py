from src.core.layers.position_encoding import TrigonoAbsPosEnc
from src.core.blocks.bert import BERTEncoderBlock
from src.core.blocks.transformer import TransformerDecoderBlock
from .config_transformer import transformerConfig
import torch.nn as nn
import math
import torch
from typing import Optional, Tuple



class TransformerEncoder(nn.Module):
    def __init__(self, config:transformerConfig):
        super().__init__()

        self.D = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_encoding = TrigonoAbsPosEnc(config.hidden_size)
        self.embd_drop = nn.Dropout(config.embd_p_drop)

        self.blks = nn.Sequential()
        for i in range(config.num_enc_blks):
            cur_blk = BERTEncoderBlock(
                config.hidden_size,
                config.num_heads,
                config.use_bias,
                config.ffn_hidden_size,
                config.attn_p_drop,
                config.resid_p_drop,
                )
            self.blks.add_module(f'enc_blk{i+1}', cur_blk)
    
    def forward(self,
                src: torch.Tensor,
                src_valid_lens: torch.Tensor
                ):
        # src shape: (batch_size, src_context_size)int64
        # src_valid_lens: (batch_size,)int64

        src_context_size = src.size(1)
         # 使用 固定位置编码时, 避免位置编码的影响过大，所以放大input embeddings
        src_embd = self.token_embedding(src)*math.sqrt(self.D) # (batch_size, src_context_size, hidden_size)

        # src position embedding: 没有 bos, 从 1 开始到 src_context_size
        position_ids = 1 + torch.arange(0, src_context_size, dtype=torch.int64, device=src_embd.device)
        src_embd = self.embd_drop( src_embd + self.pos_encoding(position_ids) ) # (batch_size, src_context_size, hidden_size)

        # attention mask(batch_size, src_context_size, src_context_size): True --> valid area, False --> need masked
        # 没有因果自回归, 只有 invalid area(PAD) --> False, valid area(non-PAD) --> True
        src_attn_mask = position_ids[None, :] < src_valid_lens[:, None] # (batch_size, src_context_size)
        src_attn_mask = src_attn_mask.unsqueeze(-1) * src_attn_mask.unsqueeze(-2)

        for blk in self.blks:
            src_embd = blk(src_embd, src_attn_mask)

        return src_embd, src_attn_mask # (batch_size, src_context_size, hidden_size) & (batch_size, src_context_size, src_context_size)
    



class TransformerDecoder(nn.Module):
    def __init__(self, config:transformerConfig):
        super().__init__()

        self.D = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_encoding = TrigonoAbsPosEnc(config.hidden_size)
        self.embd_drop = nn.Dropout(config.embd_p_drop)

        self.blks = nn.Sequential()
        for i in range(config.num_dec_blks):
            cur_blk = TransformerDecoderBlock(
                config.hidden_size,
                config.num_heads,
                config.use_bias,
                config.max_decoder_ctx_size,
                config.ffn_hidden_size,
                config.attn_p_drop,
                config.resid_p_drop,
                config.use_cached_causal_mask
                )
            self.blks.add_module(f'dec_blk{i+1}', cur_blk)
        
        # tied weight with embedding layer: 可以看作 隐藏状态和 token_embd 的语义相似度计算, 也可以视作 正则化手段防止过拟合
        self.head_tok = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.head_tok.weight = self.token_embedding.weight # 这行代码本质上将两个weight指向了同一块内存区域

        self.apply(self._init_weights) # _init_weights 中对 linear/embedding 的weights 作相同分布的初始化. 由于已tied, 故两层都以最后一次初始化为结果


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self,
                tgt: torch.Tensor,                                                      # [B, max_decoder_ctx_size]
                tgt_valid_lens: torch.Tensor,                                           # [B, ]
                encoded_output:Tuple[torch.Tensor, torch.Tensor],                       # [B, src_ctx_size, hidden_size]/[B, src_ctx_size, src_ctx_size]
                past_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                if_cache_kv: bool = False
                ):
        
        if self.training:
            positions = torch.arange(0, tgt.size(1), dtype=torch.int64, device=tgt.device) # [max_decoder_ctx_size, ]
        elif past_kv is None:
            positions = torch.tensor([0], dtype=torch.int64, device=tgt.device)
        else:
            positions = torch.tensor([0], dtype=torch.int64, device=tgt.device)









class TransformerDecoder(Decoder):
    def __init__(self, vocab_size, num_blk, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = TrigonoAbsPosEnc(num_hiddens)
        self.dropout = nn.Dropout(dropout)
        self.blks = nn.Sequential()
        for i in range(num_blk):
            cur_blk = TransformerDecoderBlock(i, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias)
            self.blks.add_module("decblock"+str(i), cur_blk)
        self.dense = nn.Linear(num_hiddens, vocab_size)


    def init_state(self, enc_outputs):
        # encoder returns encoded_src, src_valid_lens

        # train: src_enc_info = (src_enc, src_valid_lens): [(batch_size, num_steps, d_dim), (batch_size,)]
        # infer: src_enc_info = (src_enc, src_valid_lens): [(1, num_stepss, d_dim), (1,)]
        src_enc_info = enc_outputs

        return src_enc_info


    def forward(self, tgt_dec_input, src_enc_info, KV_Caches=None):

        tgt_dec_input_embd = self.embedding(tgt_dec_input) * math.sqrt(self.num_hiddens) 
        # 使用 固定位置编码时, 避免位置编码的影响过大，所以放大input embeddings

        # target position embedding
        # 训练模式: input 的 timesteps 是从 0(bos) 到 num_steps-1
        if self.training:
            _, num_steps = tgt_dec_input.shape
            position_ids = torch.arange(0, num_steps, dtype=torch.int64, device=tgt_dec_input.device) # (num_steps,)
        # 推理模式: 对于第i次infer, input 的 timestep 就是 i-1, 而这个信息可以从 KV_Caches 的values 中的第二个维度(dim=1)得到
        else:
            position_ids = torch.tensor([ 0 if KV_Caches == {} else KV_Caches['0'].size(1) ],
                                        dtype=torch.int64, device=tgt_dec_input.device) # (1,)

        # input embeddings + position embedding
        # shapes of train: (batch_size, num_steps, num_hiddens) +(broadcast) (num_steps, num_hiddens)
        # shapes of infer: (1, 1, num_hiddens) +(broadcast) (1, num_hiddens)
        tgt_query = self.dropout(tgt_dec_input_embd + self.pos_encoding(position_ids))

        # Decoder Block 的输入 tgt_query, src_enc_info, KV_Caches
        for blk in self.blks:
            # 循环过程中, 单次 blk 执行, 更新了 该 blk 对应的 KV_Caches 的 block-ID: tensor 的 kv对
            tgt_query, KV_Caches = blk(tgt_query, src_enc_info, KV_Caches)
        
        # 一次 infer forward 过程, KV_Caches 中的每个 key-value pair, 都被更新, 则 整个 KV_Caches 被更新

        #train: output[0] shape: (batch_size, num_steps, vocab_size)tensor of logits,  timestep 从 1 到 num_steps;
        #       output[1]: None
        #infer: output[0] shape: (1, 1, vocab_size)tensor of logits, 对于第i次infer, timestep 是 i;
        #       output[1]: dict of 
        #                      keys as block_indices
        #                      values as (1, i, d_dim) tensor, i 维 包含 timestep 0-i-1, 实际上就是 input KV_Caches 添加 timestep i-1 的 tgt_query
        return self.dense(tgt_query), KV_Caches



class Transformer(EncoderDecoder):
    ## 整体两种实现模式:
    # 1是encoder和decoder一起训练, 但分开使用, 这样transformer本身只需考虑train batch input的场景, infer的循环放在pred函数里
    # 2是encoder和decoder一起训练, 一起使用, 这样transformer要注意infer sample input的场景, infer的循环放在forward里. 即针对 net.training 的状态, 写不同的infer
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    # net_inputs_batch = (X_batch, Y_frontshift1_batch, X_valid_lens_batch),
    def forward(self, src, tgt_frontshift1, src_valid_lens):
        enc_outputs = self.encoder(src, src_valid_lens) # src_enc, src_valid_lens
        enc_info = self.decoder.init_state(enc_outputs)
  
        #train: output[0] shape: (batch_size, num_steps, vocab_size) tensor of logits, output[2]: None
        #infer: output[0] shape: (1, 1, vocab_size) tensor of logits, output[2]: dict of (1, cur_infer_step i, d_dim) tensor
        return self.decoder(tgt_frontshift1, enc_info)



__all__ = ["Transformer"]