# gpt2_minimal.py
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============== 配置 ===============
@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024         # 最大上下文窗口
    n_embd: int = 768               # 模型宽度
    n_layer: int = 12               # 层数
    n_head: int = 12                # 注意力头数
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    bias: bool = True               # 线性层是否使用 bias（GPT-2 用 True）

# =============== GELU（GPT-2 用的近似） ===============
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# =============== 多头自注意力（含因果遮罩 + 可选padding遮罩） ===============
class CausalSelfAttention(nn.Module):
    """
    形状约定：输入/输出 hidden_states: [B, S, D]；内部 q/k/v => [B, H, S, d], 这里 D 模型宽度 = H*d, d = dim_per_head
    """
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # qkv 合并线性（GPT-2 原版是分开的，这里合并更简洁）
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_drop = nn.Dropout(cfg.attn_pdrop)
        self.resid_drop = nn.Dropout(cfg.resid_pdrop)

        # 预先构造上三角因果mask（在forward里会按需裁剪至当前 S）
        # mask 形状 [1, 1, S, S] 便于广播到 [B,H,S,S], 正对角线上方(不包括对角线)的为True, 其余为False. 是为了选取True区域set to -inf
        # 以此排除在softmax之外
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(cfg.n_positions, cfg.n_positions, dtype=torch.bool), diagonal=1)[None, None, :, :],
            persistent=False,
        )

    def _shape_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (B, seq_len, D)
        B, seq_len, D = x.shape
        qkv = self.qkv(x)  # [B, seq_len, D] --> [B, seq_len, 3*D]
        # torch.split: 在 dim -1 维度 不重叠地 split D 宽度的 sub-tensor, 组成 sub-tensors tuple
        q, k, v = qkv.split(D, dim=-1) # [B, seq_len, 3*D] --split--> q[B, seq_len, D], k[B, seq_len, D], v[B, seq_len, D]

        def _reshape(t): # [B, seq_len, D] = [B, seq_len, H*d] -> [B, seq_len, H, d] -> [B, H, seq_len, d]
            return t.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        return _reshape(q), _reshape(k), _reshape(v) # qkv shape: [B, H, seq_len, d], 不连续

    def forward(
        self,
        x: torch.Tensor,                                               # [B, S, D]
        attention_mask: Optional[torch.Tensor] = None,                 # [B, S], 1=非PAD, 0=PAD（可选）
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (k_cache,v_cache) 形状 [B,H,cached_len,d]
        use_cache: bool = False,
    ):
        # x shape:
        B, q_len, D = x.shape # train: (B, S, D), infer: (1, 1, D)

        # 如果是 train mode, attention是 self-attention:
        #   qkv 是序列长度为 q_len=S 的 x 的不同线性映射, q=W_q(x), k=W_k(x), v=W_v(x)  -->  S(q,k) * v --> output
        #   时间步 0<->S-1 统一teacher-force预测下一个token 即 时间步 1<->S 的tokens
        q, k, v = self._shape_qkv(x)  # qkv都是 [B,H,S=q_len,d]

        # 如果是 infer mode, 即开启缓存, attention是自回归:
        #   q是序列长度为 1 的 新单步token 的线性映射, kv是 新单步token 的线性映射 追加到 前面token映射cache
        #   q: 时间步 T(0<=T<=S-1) 迭代地预测下一个 T+1 时间步token, kv: 时间步 0<->T-1 tokens cache + 时间步 T token
        #       在旧的encoder-decoder架构上, infer时, q时间步是从 0 开始的. 但是 decoder-only架构, 由于存在prompt, 那么prompt tokens
        #       decoded 就是 0<->T 时间步, q 作为时间步 T token decoded 输入, kv 是时间步 0<->T tokens decoded，从而预测时间步 T+1
        #       计prompt的最后一个token作为生成的starting query, 它的时间步是T. 那么
        #           第一次生成使用 query = 时间步T token, kv_cache = 时间步0<->T-1 token decoded, 追加后 kv_len = T+1
        #           最后一次生成使用 query = 时间步S-1 token, kv_cache = 时间步0<->S-2 token decoded, 追加后 kv_len = S
        #   记 kv_len = output_final_step: 它的值确实等于输出序列的最终时间步
        if kv_cache is not None:
            k_cache, v_cache = kv_cache  # [B,H,cached_len,d]
            k = torch.cat([k_cache, k], dim=2) # [B,H,kv_len,d], 这里 kv_len = cached_len + 1, starting query timestep < kv_len <= S
            v = torch.cat([v_cache, v], dim=2) # [B,H,kv_len,d]

        # 注意力分数
        # train: q[B,H,S,d] @ k[B,H,S,d].transpose(-2,-1) -> q[B,H,S,d] @ k[B,H,d,S] -> att[B,H,S,S]
        # infer: q[1,H,1,d] @ k[1,H,kv_len,d].transpose(-2,-1) -> q[1,H,1,d] @ k[1,H,d,kv_len] -> att[1,H,1,kv_len]
        # 总结: q[B,H,q_len,d] @ k[B,H,kv_len,d].transpose(-2,-1) -> [B,H,q_len,kv_len]
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,q_len,kv_len]
        # output_final_step = S for train mode; output_final_step = input query 时间步 + 1 = output query 时间步 for infer mode
        output_final_step = att.size(-1)

        # 因果遮罩：禁止看未来位置
        # casual_mask 是 [1,1,S,S], dim-2代表query时间步0<->S-1，dim-1代表output时间步1<->S: 对于query i-1, output时间步 i<->S 都是未来

        # train mode: qkv都是 从x线性映射而来的长度为 S 的序列, casual_mask本身就代表了对每一个q而言的未来mask
        # infer mode: q是时间步为T的单步token映射, T+1=kv_len=output_final_step; kv是时间步0<->T的tokens decoded, casual_mask应该是T代表的那一行

        # 所以 att [B,H, q_len, output_final_step] 的 dim-2也表示query的时间步, dim-1是对应query的输出最终时间步.
        # 在 casual_mask 上截取 dim-2 维度q_len长，dim-1 维度最终输出时间步为kv_len 的mask
        causal = self.causal_mask[:, :, output_final_step-q_len:output_final_step, :output_final_step] # [1, 1, q_len, kv_len]
        
        att = att.masked_fill(causal, float("-inf")) # [B,H,q_len,kv_len]

        # 可选的 padding mask：在 label data 中, 代表屏蔽 pad 位置. 1代表非pad位置, 0代表pad位置
        # attention_mask shape: [B, output_final_step], 是 label data 的 shape
        if attention_mask is not None:
            # attention_mask: [B, output_final_step]（当有cache时 output_final_step = cached_len + 1）
            # 转为 key padding mask: True 表示需要屏蔽
            key_pad = (attention_mask == 0).view(B, 1, 1, output_final_step) 
            att = att.masked_fill(key_pad, float("-inf"))

        att = F.softmax(att, dim=-1) # [B,H,q_len,kv_len]
        att = self.attn_drop(att) # [B,H,q_len,kv_len]

        y = torch.matmul(att, v)  # [B, H, q_len, kv_len] @ [B, H, kv_len, d] -> [B, H, q_len, d]
        y = y.transpose(1, 2).contiguous().view(B, q_len, D)  # [B, H, q_len, d] -> [B, q_len, H, d] -> [B, q_len, D=H*d]
        y = self.resid_drop(self.proj(y))                 # 输出投影 + dropout

        new_cache = None
        if use_cache:
            # 返回完整的 K/V 以供下一步复用
            new_cache = (k, v)

        return y, new_cache

# =============== MLP ===============
class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.fc_in = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.act = GELU()
        self.fc_out = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.act(x)
        x = self.fc_out(x)
        x = self.drop(x)
        return x

# =============== Block（Pre-LN） ===============
class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.mlp = MLP(cfg)

    def forward(self, x, attention_mask=None, kv_cache=None, use_cache=False):
        # Self-Attention
        sa_out, new_cache = self.attn(self.ln_1(x), attention_mask=attention_mask, kv_cache=kv_cache, use_cache=use_cache)
        x = x + sa_out
        # MLP
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out
        return x, new_cache

# =============== GPT-2 Model（带LM头） ===============
class GPT2LMHeadModel(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token embedding
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)  # position embedding（GPT-2 原版）
        self.drop = nn.Dropout(cfg.embd_pdrop)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # 权重 tying：lm_head 与 token embedding 共享
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,                 # [B, S]
        attention_mask: Optional[torch.Tensor] = None,  # [B, S], 1=非PAD, 0=PAD
        use_cache: bool = False,
        past_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,  # 每层的(k,v)
    ):
        B, S = input_ids.shape
        device = input_ids.device

        # 位置索引（GPT-2 使用绝对位置，从 0..S_total-1）
        if past_kv is None:
            past_len = 0
        else:
            past_len = past_kv[0][0].size(2)  # 取第一层的缓存长度
        pos_ids = torch.arange(past_len, past_len + S, device=device).unsqueeze(0)  # [1,S]

        # 嵌入
        tok = self.wte(input_ids)           # [B,S,C]
        pos = self.wpe(pos_ids)             # [1,S,C]（广播到B）
        x = self.drop(tok + pos)            # [B,S,C]

        # 若存在 cache，attention_mask 也要扩展到 past_len+S 的长度
        # 这里简单处理：若传入的是当前S的mask，且有cache，则把前缀视为全1（非PAD）
        if attention_mask is not None and past_len > 0:
            pad = torch.ones(B, past_len, dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([pad, attention_mask], dim=1)  # [B, past_len+S]

        new_past = [] if use_cache else None

        # 逐层
        for i, block in enumerate(self.h):
            kv = None if past_kv is None else past_kv[i]
            x, new_kv = block(x, attention_mask=attention_mask, kv_cache=kv, use_cache=use_cache)
            if use_cache:
                new_past.append(new_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B,S,vocab]

        return logits, (tuple(new_past) if use_cache else None)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,                # [B, S]
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
    ):
        """
        简单的自回归生成（带 KV cache）
        """
        self.eval()
        B, S = input_ids.shape
        device = input_ids.device

        past_kv = None
        attention_mask = torch.ones(B, S, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits, past_kv = self(
                input_ids[:, -1:].contiguous() if past_kv is not None else input_ids,
                attention_mask=attention_mask if past_kv is None else torch.ones(B, 1, device=device, dtype=torch.long),
                use_cache=True,
                past_kv=past_kv,
            )
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # [B,vocab]

            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k, dim=-1)
                kth = values[..., -1, None]
                logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B,1]
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(B, 1, device=device, dtype=torch.long)], dim=1)

            if eos_id is not None:
                # 若所有样本都已结束则提前退出
                if (next_token == eos_id).all():
                    break

        return input_ids

# =============== 使用示例（最小跑通） ===============
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GPT2Config(
        vocab_size=1000, n_positions=128, n_embd=128, n_layer=4, n_head=4,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, bias=True
    )
    model = GPT2LMHeadModel(cfg)

    B, S = 2, 16
    x = torch.randint(0, cfg.vocab_size, (B, S))
    mask = torch.ones(B, S, dtype=torch.long)
    logits, _ = model(x, attention_mask=mask)
    print("logits:", logits.shape)  # [2,16,1000]

    # 简单生成
    out = model.generate(x[:, :8], max_new_tokens=8, temperature=1.0, top_k=50, eos_id=None)
    print("generated:", out.shape)  # [2, 16]

