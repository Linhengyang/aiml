from ...core.nn_components.meta_frames import Decoder, DecoderOnly
from ...core.nn_components.root_layers.position_encoding import LearnAbsPosEnc
from ...core.nn_components.sub_modules._gpt2 import GPT2DecoderBlock
import torch.nn as nn
import math
import torch
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class GPT2Config:
    ## embedding layer configs
    embd_size:int
    vocab_size:int
    embd_p_drop:float
    ## decoder-block(casual_attention layer + ffn) configs
    # embd_size:int
    num_head:int
    use_bias:bool
    max_context_size:int
    attn_p_drop:float
    resid_p_drop:float
    use_cached_casual_mask:bool
    use_rope:bool
    ## number of decoder-block
    num_block:int



class gpt2(DecoderOnly):
    '''
    Decoder-Only 的 GPT2 架构:

                                               seq[B, S] --embd--> tok[B, S, D]|
    (if seq[B, S] use ABS as pos_enc) position_ids[1, S]|--embd--> pos[1, S, D]|--drop--> x[B, S, D]|---->
                                                                              (if need past) past_kv|
                                       (if need past) past_attention_mask|--concat--> attention_mask|
                (if seq[B, S] involves PAD elements) attention_mask[B, S]|

    >----decoder_blocks|--> x[B, S, D] -------layer_norm--reverse_embd-------> logits[B, S, vocab_size]
                       |--> new_kv(if need) -----append_to_new_container-----> new_kv_caches(if need)
                       |-----------------------------------------------------> new_past_attention_mask(if need)
    '''
    def __init__(self, config:GPT2Config):
        super().__init__()
        self.config = config

        self.W_tok_embd = nn.Embedding(config.vocab_size, config.embd_size)
        if not config.use_rope:
            self.W_pos_embd = LearnAbsPosEnc(config.max_context_size, config.embd_size) # 可学习绝对位置编码自带随机初始化
        self.embd_drop = nn.Dropout(config.embd_p_drop)

        self.blocks = nn.ModuleList([
            GPT2DecoderBlock(
                config.embd_size,
                config.num_head,
                config.use_bias,
                config.max_context_size,
                config.attn_p_drop,
                config.resid_p_drop,
                config.use_cached_casual_mask,
                config.use_rope) for _ in range(config.num_block)])
        
        self.layer_norm_final = nn.LayerNorm(config.embd_size)

        # tied weight with embedding layer: 可以看作 隐藏状态和 token_embd 的语义相似度计算, 也可以视作 正则化手段防止过拟合
        self.head_tok = nn.Linear(config.embd_size, config.vocab_size, bias=False)
        self.head_tok.weight = self.W_tok_embd.weight # (vocab_size, embd_size). 这行代码本质上将两个weight指向了同一块内存区域

        self.apply(self._init_weights) # _init_weights 中对 linear/embedding 的weights 作相同分布的初始化. 由于已tied, 故两层都以最后一次初始化为结果

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,
                input_seqs: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None, # describe input_seqs, [B, S]
                past_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None, # tuple of (past_k, past_v)[B, H, num_steps_past, d]
                past_attention_mask: Optional[torch.Tensor] = None, # describe past_kv, [B, num_steps_past]
                if_cache_kv: bool = False # if cache the new updated kv or not 
                ):
        '''
        input:
            input_seqs: tensor [B, S]
            attention_mask: None or tensor [B, S] to describe whether input_seqs PAD
            past_kv: None or tuple of (k_cache, v_cache) where k_cache/v_cache [B, S, num_steps_past, d]
            past_attention_mask: None or tensor [B, num_steps_past] to describe whether past_kv PAD
            if_cache_kv: False or True to determine if updated k_cache/v_cache returned in tuple

        output:
            logits: [B, S, vocab_size]
            past_kv: None or tuple of updated k_cache/v_cache
            past_attention_mask: None or updated past_attention_mask which stick to past_kv
        '''
        B, S = input_seqs.shape
        device = input_seqs.device

        tok = self.W_tok_embd(input_seqs) # [B, S] -> [B, S, D]
        # 如果存在绝对位置编码层: 要 add abs pos encoding 到 tok embedding 上.
        if hasattr(self, 'W_pos_embd'):
            if past_kv is not None:
                start_position = past_kv[0][0].size(2)
            else:
                start_position = 0
            # train/prefill: 0<->S-1; decode(此时S=1): num_steps_past
            position_ids = torch.arange(start_position, start_position + S, device=device).unsqueeze(0) # [1, S]
            pos = self.W_pos_embd(position_ids) # [1, S, D]
            tok = tok + pos # add abs pos encoding 到 tok embedding 上
        # 如果使用RoPE位置编码: 无需额外操作, casual attention 层会执行RoPE
        x = self.embd_drop(tok) # [B, S, D]

        if past_kv is not None: # attention_mask 作为对 input_seq 的描述, 只有当 past_kv not None 才不等价于对 v 的描述, 才可能需要更新它
            if past_attention_mask is not None or attention_mask is not None:
                # 若 past_attention_mask 和 attention_mask 有其一不为 None, 那么说明需要引入 pad 信息
                if past_attention_mask is None:
                    past_attention_mask = torch.ones(B, past_kv[0][1].size(2))
                if attention_mask is None:
                    attention_mask = torch.ones(B, S)
                attention_mask = torch.cat([past_attention_mask, attention_mask], dim=-1).to(device=device) # [B, num_steps_past+1]
            # 若 past_attention_mask 和 attention_mask 都是 None, 那么 无需更新 attention_mask: 以None输入即可
        
        new_past_kv = [] if if_cache_kv else None

        for i, block in enumerate(self.blocks):
            kv_cache = past_kv[i] if past_kv is not None else None
            x, new_kv_cache = block(x, kv_cache, attention_mask, if_cache_kv)
            if if_cache_kv:
                new_past_kv.append( new_kv_cache )
        
        logits = self.head_tok(self.layer_norm_final(x)) # [B, S, vocab_size]

        return logits, tuple(new_past_kv) if if_cache_kv else None, attention_mask