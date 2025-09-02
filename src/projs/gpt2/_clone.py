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
        # x shape: (B, S, D)
        B, S, D = x.shape
        qkv = self.qkv(x)  # [B, S, D] --> [B, S, 3*D]
        # torch.split: 在 dim -1 维度 不重叠地 split D 宽度的 sub-tensor, 组成 sub-tensors tuple
        q, k, v = qkv.split(D, dim=-1) # [B, S, 3*D] --split--> q[B, S, D], k[B, S, D], v[B, S, D]

        def _reshape(t): # [B,S,D] = [B, S, H*d] -> [B, S, H, d] -> [B, H, S, d]
            return t.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        
        return _reshape(q), _reshape(k), _reshape(v) # q[B, H, S, d], k[B, H, S, d], v[B, H, S, d]

    def forward(
        self,
        x: torch.Tensor,                                               # [B, S, D]
        attention_mask: Optional[torch.Tensor] = None,                 # [B, S], 1=非PAD, 0=PAD（可选）
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (k_cache,v_cache) 形状 [B,H,S_cached,D]
        use_cache: bool = False,
    ):
        B, S, D = x.shape
        q, k, v = self._shape_qkv(x)  # [B,H,S,d]

        # 如果开启缓存（自回归解码），把新步的 k,v 追加到 cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache  # [B,H,S_cached,D]
            k = torch.cat([k_cache, k], dim=2)  # [B,H,S_total,D]
            v = torch.cat([v_cache, v], dim=2)

        # 注意力分数
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,S_q,S_k]
        S_total = att.size(-1)

        # 因果遮罩：禁止看未来位置
        causal = self.causal_mask[:, :, :S, :S_total]           # [1,1,S,S_total]
        att = att.masked_fill(causal, float("-inf"))

        # 可选的 padding mask：屏蔽 encoder 序列中的 PAD（这里是 decoder 自注意力的 padding）
        if attention_mask is not None:
            # attention_mask: [B,S_total]（当有cache时 S_total=S_cached+S_new）
            # 转为 key padding mask: True 表示需要屏蔽
            key_pad = (attention_mask == 0).view(B, 1, 1, S_total)  # [B,1,1,S_total]
            att = att.masked_fill(key_pad, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.matmul(att, v)  # [B,H,S,D]
        y = y.transpose(1, 2).contiguous().view(B, S, C)  # [B,S,C]
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
