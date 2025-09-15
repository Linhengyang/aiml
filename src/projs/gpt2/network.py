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
            attention_mask: None or tensor [B, S] to describe input_seqs PAD-info. False-->PAD, True-->non-PAD
            past_kv: None or tuple of (k_cache, v_cache) where k_cache/v_cache [B, S, num_steps_past, d]
            past_attention_mask: None or tensor [B, num_steps_past] to describe past_kv PAD-info
            if_cache_kv: bool to determine if updated k_cache/v_cache returned in tuple

        output:
            logits: [B, S, vocab_size]
            past_kv: None or tuple of updated k_cache/v_cache(so-far). for next-time, it'll be past
            past_attention_mask: None or updated past_attention_mask which stick to past_kv(so-far). for next-time, it'll be past
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
                    past_attention_mask = torch.ones(B, past_kv[0][1].size(2), dtype=torch.bool, device=device)
                if attention_mask is None:
                    attention_mask = torch.ones(B, S, dtype=torch.bool, device=device)
                attention_mask = torch.cat([past_attention_mask, attention_mask], dim=-1).to(device=device) # [B, num_steps_past+1]
            # 若 past_attention_mask 和 attention_mask 都是 None, 那么 无需更新 attention_mask: 以None输入即可
        
        new_past_kv = [] if if_cache_kv else None

        for i, block in enumerate(self.blocks):
            kv_cache = past_kv[i] if past_kv is not None else None
            x, new_kv_cache = block(x, kv_cache, attention_mask, if_cache_kv)
            if if_cache_kv:
                new_past_kv.append( new_kv_cache ) # new_kv_cache 是 torch.cat 得到的, 其内存使用是高效的.
        
        logits = self.head_tok(self.layer_norm_final(x)) # [B, S, vocab_size]

        return logits, tuple(new_past_kv) if if_cache_kv else None, attention_mask
    
    @staticmethod
    def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int|None):
        # logits: [B, vocab_size]
        logits /= max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            # values shape: [B, top_k]
            values, _ = torch.topk(logits, top_k, dim=-1) # 在 logits[B, n_vocab] dim-1 取每行的 top_k, 从大到小 从左至右排列
            # mask for logits to remove all elements outside tok_k
            kth = values[..., -1, None] # [B, top_k] -> [B, 1]
            # where outside top_k --> -inf; where inside top_k --> logits
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits) # [B, vocab_size]
        
        probs = torch.nn.functional.softmax(logits, dim=-1) # [B, vocab_size]
        # select 1 via every row distribution as next-token
        next_token = torch.multinomial(probs, num_samples=1, replacement=False)  # [B, 1]

        return next_token

    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor, # prefill: [B, S], decode: [B, 1]. latest timestep --> T >= 0
                 max_gen_size: int,        # max generating length
                 temperature: float = 1.0, # flatten/sharpen the output distribution
                 top_k: int|None = None,   # limit selections when sampling token via output distribution
                 eos_id: int|None = None   # stop sign
                 ):
        '''
        generate 分为两个阶段:
        1. prefill: input_ids [B, S], latest timestep T
        此时, forward 输入:
            input_seqs = input_ids [B, S], latest timestep T
            attention_mask 根据 input_seqs 的实际 PAD 情况输入, None/[B, S]
            past_kv = None 因为此时没有 past
            past_attention_mask = None 因为此时没有 past
            if_cache_kv = True 因为在 generate 时, 总需要把该次 so-far kv 作为下一次 infer 的 past_kv 输入

        prefill 的输出 output [B, S], latest timestep T+1. 取出 output 的 last timestep 组成 [B, 1] 输入进入阶段2
        2. decode: input_ids [B, 1], latest timestep T+1
        此时, forward 输入:
            input_seqs = input_ids [B, 1], latest timestep T+1
            attention_mask 根据 input_seqs 的实际 PAD 情况输入. 考察 input_ids 中是否是 eos_id 决定 PAD, None/[B, 1]
            past_kv = tuple of kv pair, past 指 0<->T timesteps, 即上一次 generate 返回的 so-far kv. 上一次的so-far是这一次的past
            past_attention_mask: None/描述 past v 的 TF tensor, 即上一次 generate 返回的 so-far attn_mask. 上一次的so-far是这一次的past
            if_cache_kv = True 因为在 generate 时还是要保持输出该次 so-far kv 作为下一次 infer 的 past_kv 输入
        '''
        assert max_gen_size > 0, f'max_gen_size must be larger than 0'
        self.eval()

        # prefill
        attention_mask = None
        if eos_id is not None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool) # same device with input_ids

        logits, past_kv, past_attention_mask = self(
            input_seqs = input_ids,             # timesteps 0<->T
            attention_mask = attention_mask,    # timesteps 0<->T
            past_kv = None,
            past_attention_mask = None,
            if_cache_kv = True
        )
        # returned logits: [0, timesteps 1<->T+1, vocab_size]
        # past_kv: tuple of kv_cache [B, H, timesteps 0<->T, d]
        # past_attention_mask: describe of past v [B, timesteps 0<->T]

        max_gen_size -= 1
        next_token = self.sample_next_token(logits[:, -1, :], temperature, top_k) # [B, 1], timestep T+1

        attention_mask = None
        # 若 eos_id 不为 None, 则提供了判断 PAD 的标准: 等于 eos_id 的就是 PAD 位置.
        if eos_id is not None:
            attention_mask = next_token != eos_id # [B, 1]
            if not attention_mask.any(): # 如果生成的 next_token 全都是 PAD --> 结束 generate
                return next_token
        
        # decode
        for _ in range(max_gen_size):
            logits, past_kv, past_attention_mask = self(
                input_seqs = next_token,                    # [B, 1], latest timestep T+1
                attention_mask = attention_mask,            # [B, 1]/None
                past_kv = past_kv,                          # tuple of kv_cache [B, timesteps 0<->T]
                past_attention_mask = past_attention_mask,
                if_cache_kv = True
            )