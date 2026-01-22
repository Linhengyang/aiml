from ...core.architectures import Decoder, DecoderOnly
from ...core.nn_components.root_layers.position_encoding import LearnAbsPosEnc
from ...core.nn_components.sub_modules._gpt2 import GPT2DecoderBlock
from .function import get_segs_pos_attnmask_train, get_segs_pos_attnmask_prefill, get_segs_pos_attnmask_decode
import torch.nn as nn
import math
import torch
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class gpt2Config:
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
    def __init__(self, config:gpt2Config):
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
                input_segs: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                past_segs: Optional[torch.Tensor] = None,
                if_cache_kv: bool = False
                ):
        '''
        input:
            input_seqs: tensor [B, L_q]long whose value must within embedding table
            input_segs: None | tensor [B, L_q]long to describe input_seqs' PAD-info(0 for PAD) & TEXT-info(1/2...for textID)
            past_kv: None | tuple of past(k_cache, v_cache), k_cache/v_cache tensor [B, H, L_past, d]float
            past_segs: None | tensr [B, L_past]long to describe past_kv's PAD-info & TEXT-info
            if_cache_kv: bool to determine if to return updated k_cache/v_cache as in tuple

        output:
            logits: [B, L_q, vocab_size]
            past_kv: None | tuple of so_far(k_cache, v_cache). for next-time, it'll be past
        
        本模型默认准则: segment 中, 0 代表 PAD, PAD 的 segment 赋 0; 位置编码中, PAD 位置编码赋 0, 有效序列的位置编码从 0 开始(第一位赋0)
        本模型对 input 序列中, PAD 具体赋哪个值 并不关心(因为 PAD 会被 attention_mask 屏蔽计算). 但是此值必须在embedding table范围内.
        数据处理时, input 序列中的 PAD 位置可用 0 直接填充, 这样永远不会 out-of-table, 处理也方便.
        '''
        B, L_q = input_seqs.shape
        device = input_seqs.device
        L_past = past_kv[0][0].size(2) if past_kv is not None else 0

        if self.training:
            segments, positions, attention_mask = get_segs_pos_attnmask_train(input_segs, L_q, device=device)
        elif past_kv is None:
            segments, positions, attention_mask = get_segs_pos_attnmask_prefill(input_segs, L_q, device=device)
        else:
            segments, positions, attention_mask = get_segs_pos_attnmask_decode(input_segs, L_q, past_segs, L_past, device=device)
        
        tok = self.W_tok_embd(input_seqs) # [B, L_q] -> [B, L_q, D]

        # 如果存在绝对位置编码层: 要 add abs pos encoding 到 tok embedding 上
        if hasattr(self, 'W_pos_embd'):
             # positions [B, L_so_far=L_past + L_q], 取 last L_q 列
            positions_q = positions[:, -L_q:] # [B, L_so_far] --> [B, L_q]
            
            tok = tok + self.W_pos_embd(positions_q) # [B, L_q, D]
            positions = None # 位置编码已经加到 tok embedding 里了, 不再使用
        
        # 如果使用RoPE位置编码: 无需额外操作, casual attention 层会执行RoPE
        x = self.embd_drop(tok) # [B, L_q, D]
        
        new_past_kv = [] if if_cache_kv else None
        for i, block in enumerate(self.blocks):
            kv_cache = past_kv[i] if past_kv is not None else None
            x, new_kv_cache = block(x, kv_cache, if_cache_kv, attention_mask, positions)
            if if_cache_kv:
                new_past_kv.append( new_kv_cache ) # new_kv_cache 是 torch.cat 得到的, 其内存使用是高效的.
        
        logits = self.head_tok(self.layer_norm_final(x)) # [B, L_q, vocab_size]

        return logits, tuple(new_past_kv) if if_cache_kv else None, segments
    

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
                 input_ids: torch.Tensor, # prefill: [B, L_q], decode: [B, 1]
                 input_segs: torch.Tensor|None, # [B, L_q]
                 max_gen_size: int,        # max generating length
                 temperature: float = 1.0, # flatten/sharpen the output distribution
                 top_k: int|None = None,   # limit selections when sampling token via output distribution
                 eos_id: int|None = None   # stop sign
                 ):
        '''
        generate 分为两个阶段:
        1. prefill: input_ids [B, S]
        此时, forward 输入:
            input_seqs = input_ids [B, S]
            input_segs 根据 input_seqs 的实际 PAD/TEXT 情况输入, None/[B, S]. 实际中 infer 中不再屏蔽跨文档相关性, 仅屏蔽PAD
            past_kv = None 因为此时没有 past
            past_segs = None 因为此时没有 past
            if_cache_kv = True 因为在 generate 时, 总需要把该次 so-far kv 作为下一次 infer 的 past_kv 输入
        prefill 的输出 output [B, S]. 取出 output 的 last dim 2 --> [B, 1] 作为输入进入阶段2

        2. decode: input_ids [B, 1]
        此时, forward 输入:
            input_seqs = input_ids [B, 1]
            input_segs = None 即使产出 eos 也不算 PAD, 希望模型考虑 eos 而不是屏蔽 eos
            past_kv = tuple of k/v [B, H, L_past, d], 即上一次 generate 返回的 so-far kv. 上一次的so-far是这一次的past
            past_segs: None/[B, L_past], 即上一次 generate 返回的 so-far segments. 上一次的so-far是这一次的past
            if_cache_kv = True 因为在 generate 时还是要保持输出该次 so-far kv 作为下一次 infer 的 past_kv 输入
        '''
        assert max_gen_size > 0 and max_gen_size + input_ids.size(1) <= self.config.max_context_size, \
            f'max_gen_size must be larger than 0 and not larger than max_context_size minus prompt sequence length'
        self.eval()

        # prefill
        logits, past_kv, past_segs = self(
            input_seqs = input_ids,             # [B, L_q]
            input_segs = input_segs,            # [B, L_q]|None
            past_kv = None,
            past_segs = None,
            if_cache_kv = True
        )
        # logits: [B, S, vocab_size]
        # past_kv: tuple of kv_cache [B, H, L_past=L_q, d]
        # past_segs: describe of past kv [B, L_past=L_q]|None

        max_gen_size -= 1
        next_token = self.sample_next_token(logits[:, -1, :], temperature, top_k) # [B, 1]
        output = next_token.cpu() # [B, 1]

        # 若提供了 EOS, 则检查输出结果是否全 EOS. 若是, 则退出 generate
        if eos_id is not None and (next_token == eos_id).all():
            return output
        
        # decode
        for _ in range(max_gen_size):
            logits, past_kv, past_segs = self(
                input_seqs = next_token,                    # [B, 1]
                input_segs = None,                          # None
                past_kv = past_kv,                          # tuple of kv_cache [B, H, L_past, d]
                past_segs = past_segs,                      # [B, L_past]|None
                if_cache_kv = True)

            # logits: [B, 1, vocab_size]
            # past_kv: tuple of kv_cache [B, H, L_past+1, d]
            # past_segs: describe of past kv [B, L_past+1] | None
            next_token = self.sample_next_token(logits.squeeze(1), temperature, top_k) # [B, 1]
            output = torch.cat([output, next_token.cpu()], dim=-1)

            # 若提供了 EOS, 则检查输出结果是否全 EOS. 若是, 则退出 generate
            if eos_id is not None and (next_token == eos_id).all():
                return output
        
        return output # [B, max_gen_size]