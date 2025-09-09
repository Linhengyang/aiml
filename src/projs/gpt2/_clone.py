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

        # 预先构造上三角因果 casual mask（在forward里会按需裁剪 L_max, L_max 至当前所需）
        # casual mask 形状 [1, 1, L_max, L_max]. 长度 L_max 分别代表时间步 0<->L_max-1, 正对角线上方(不包括对角线)的为True, 其余为False. 
        # True 部分代表了对每一个时间步 T (0<=T<=L_max-1)而言，哪些时间步是未来（True部分），哪些是现在及过去（False部分）
        # 后两维直接拉满到L_max是为了方便裁剪, 前两位松弛到1是为了方便广播

        # next推理的基本哲学：token_of_step_T+1 = f{token_of_step_T, ..., token_of_step_0}, 姑且认为这个f是线性函数: 线性自回归
        # 考虑 num_steps_q 次next-token预测，那么 num_steps_q 是需要next推理的时间步个数, 是自由设定的。
        # 令其中最晚的时间步为 T。那么需要准备如下：
        #   1. 需要 v[..., T+1, d] 代表 时间步 0<->T 的tokens。
        #   2. 需要确认各时间步的自回归系数矩阵 attention
        #      attention 是 qk 计算得到的, q[..., num_steps_q, d], k[..., num_steps_kv, d] --> [..., num_steps_q, num_steps_kv]
        # 这里 num_steps_kv 是对next推理的时间步的总包络时间长度，也就是 T+1（即0<->T，因为根据基本哲学，需要包络0至T所有，才能推理T+1）
        # 后文记 num_steps_kv 为 num_steps_so_far

        # 所以 attention [..., num_steps_q(latest timestep T), num_steps_so_far=T+1(timesteps 0 to T)] 就是 num_steps_q 次next推理的自回归系数
        # 对每一次next推理而言, 考虑它的时间步是 t（0 <= t <= T）, 那么它的自回归因果遮罩
        #   就是casual mask取时间步t对应的行, 取0至T时间步列, 共 num_steps_q（latest_timestep T） 行，num_steps_so_far=T+1 列
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(cfg.n_positions, cfg.n_positions, dtype=torch.bool), diagonal=1)[None, None, :, :],
            persistent=False,
        )
        # 因果遮罩的True部分，是每个next推理的"未来", 要去掉它们对next token的影响，所以它们对 v 的权重赋0，即softmax前打分赋 -inf

    def _shape_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (B, num_steps_q, D)
        B, num_steps_q, D = x.shape
        qkv = self.qkv(x)  # [B, num_steps_q, D] --> [B, num_steps_q, 3*D]
        # torch.split: 在 dim -1 维度 不重叠地 split D 宽度的 sub-tensor, 组成 sub-tensors tuple
        q, k, v = qkv.split(D, dim=-1) # [B, num_steps_q, 3*D] --split--> qkv 都是 [B, num_steps_q, D]

        def _reshape(t): # [B, num_steps_q, D] = [B, num_steps_q, H*d] -> [B, num_steps_q, H, d] -> [B, H, num_steps_q, d]
            return t.view(B, num_steps_q, self.n_head, self.head_dim).transpose(1, 2)
        return _reshape(q), _reshape(k), _reshape(v) # qkv shape: [B, H, num_steps_q, d], 不连续

    def forward(
        self,
        x: torch.Tensor,                                               # [B, num_steps_for_query, D], latest timestep T
        attention_mask: Optional[torch.Tensor] = None,                 # [B, num_steps_so_far], num_steps_so_far=T+1, 1=非PAD, 0=PAD（可选）
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (k_cache,v_cache) 形状 [B, H, num_steps_past, d]
                                                                       #    num_steps_past + 1 = num_steps_so_far
        use_cache: bool = False,
    ):
        # x shape:
        B, num_steps_for_query, D = x.shape # train: (B, S, D), infer: (B, 1, D)

        # 如果是 train mode:
        #   qkv 是序列长度为 S 的 x 的不同线性映射, q=W_q(x), k=W_k(x), v=W_v(x)  -->  S(q,k) * v --> output
        #   时间步 0<->S-1 统一teacher-force预测下一个token 即 时间步 1<->S 的tokens
        q, k, v = self._shape_qkv(x)  # qkv都是 [B, H, S, d]

        # 如果是 infer mode, 即开启缓存:
        #   q是序列长度为 1 的 新单步token tensor, 记该单步为时间步 T（0 <= T < L_max），kv_cached是 past(cached) tokens tensor，时间步是 0至T-1
        #   q: 时间步 T 迭代地预测next token 即 T+1 时间步, kv: past（时间步0<->T-1）tokens + 当前（时间步 T） token，得到 0<->T tokens tensor
        #       在旧的encoder-decoder架构上, infer时, q时间步是从 0 开始的. 但是 decoder-only架构, 由于存在prompt, 那么prompt tokens
        #       decoded 就是 0<->T 时间步, q 作为时间步 T token decoded 输入, kv 是时间步 0<->T tokens decoded，从而预测时间步 T+1
        #       记 prompt的最后一个token作为生成的starting query, 它的时间步是T. 那么
        #       第一次生成使用 query = 时间步T token, kv_cache = past（时间步0<->T-1） tokens tensor, 追加当前（时间步T）后 kv_len = T+1
        #       最后一次生成使用 query = 时间步L_max-1 token, kv_cache = past（时间步0<->L_max-2） tokens tensor, 追加当前后 kv_len = L_max

        #   记 T = final_step_of_query, (0 <= T < L_max), 则 num_steps_so_far = T+1（0至T）
        #   训练时的query：作 T+1 次next token预测：num_steps_for_query = T+1，即对 时间步0 至 时间步T 都作next token预测.
        #       这里 latest timestep = T. 故 num_steps_kv 作为总包络时间长度 = num_steps_so_far = T+1，则 kv = 0至T所有时间步token tensor
        #       从而 q = k = v = 0至T所有时间步token tensor. 相应因果遮罩 行取 时间步 0至T，列取 时间步 0至T
        #   推理时的query：作 1 次 next token 预测：num_steps_for_query = 1，即对 时间步T 作next token预测 时间步T+1
        #       这里 latest timestep = T. 故 num_steps_kv 作为总包络时间长度 = num_steps_so_far = T+1，则 kv = 0至T所有时间步token tensor
        #       由于时间步 T token tensor 作为query输入, 那么还需要 past（0<->T-1）token tensor 作为 kv_cached 输入
        #       从而 q = 单时间步 T 的token tensor, k/v_cached = num_steps_past（0至T-1）的tokens tensor
        #       kv = kv_cached + q = num_steps_so_far=T+1（0至T）的tokens tensor. 相应因果遮罩 行取 时间步T，列取 时间步 0至T
        if kv_cache is not None:
            k_cache, v_cache = kv_cache  # [B, H, num_steps_past=T, d], kv_cache 的 num_steps_past 代表 0<->T-1 
            k = torch.cat([k_cache, k], dim=2) # [B, H, num_steps_so_far=T+1, d]
            v = torch.cat([v_cache, v], dim=2) # [B, H, num_steps_so_far=T+1, d]

        # attention 自回归系数矩阵
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, num_steps_for_query=S或1, num_steps_so_far=T+1]

        num_steps_so_far = att.size(-1) # num_steps_so_far = latest_timestep_in_query + 1 = T+1

        # 对于 att [B, H, num_steps_for_query=S或1, num_steps_so_far=T+1], 考虑其 因果遮罩
        # train mode: q = k = v = 0至T所有时间步token tensor，相应因果遮罩：根据 q 行取 时间步 0至T，根据 k 列取 时间步 0至T
        # infer mode: q = 单时间步 T 的token tensor, kv = kv_cached + q = num_steps_so_far=T+1（0至T）的tokens tensor
        #             相应因果遮罩：根据 q 行取 单时间步T，根据 k 列取 时间步 0至T

        # 两个mode可以归一化：列取时间步 0至T, 共T+1 = num_steps_so_far 列，行取 num_steps_for_query 行 with 最终时间步 T=num_steps_for_query-1
        causal = self.causal_mask[:, :, num_steps_so_far-num_steps_for_query:num_steps_so_far, :num_steps_so_far]
        
        att = att.masked_fill(causal, float("-inf")) # [B, H, num_steps_for_query, num_steps_so_far]

        # 可选的 padding mask：无论是 train 还是 infer，对于 latest timestep = T 的query，都涉及到 num_steps_so_far=T+1 个位置
        # 故 attention_mask shape: [B, num_steps_so_far=T+1]，
        # 代表在 num_steps_so_far=T+1（时间步0至T）这么多位置中, 实际是 pad 的位置. 1代表非pad位置, 0代表pad位置
        if attention_mask is not None:
            # attention_mask: [B, num_steps_so_far=T+1]
            # 转为 key padding mask: True 表示需要屏蔽
            key_pad = (attention_mask == 0).view(B, 1, 1, num_steps_so_far) 
            att = att.masked_fill(key_pad, float("-inf"))

        att = F.softmax(att, dim=-1) # [B, H, num_steps_for_query=S或1, num_steps_so_far=T+1]
        att = self.attn_drop(att) # [B, H, num_steps_for_query=S或1, num_steps_so_far=T+1]

        # [B, H, num_steps_for_query=S/1, num_steps_so_far=T+1] @ [B, H, num_steps_so_far, d] -> [B, H, num_steps_for_query=S/1, d]
        y = torch.matmul(att, v)
        # [B, H, num_steps_for_query=S/1, d] -> [B, num_steps_for_query=S/1, H, d] -> [B, num_steps_for_query=S/1, D=H*d]
        y = y.transpose(1, 2).contiguous().view(B, num_steps_for_query, D)
        y = self.resid_drop(self.proj(y))                 # 输出投影 + dropout

        new_cache = None
        if use_cache:
            # 返回完整的 K/V 以供下一步复用
            new_cache = (k, v)
        
        # output y shape: [B, num_steps_for_query=S/1, D=H*d], timestep应该统一往前一步
        # 对比一下 input x shape: [B, num_steps_for_query=S/1, D], latest timestep T
        # new_cache: None or tuple of kv # [B, H, num_steps_so_far=T+1, d]
        # 对比一下 old_cache shape：[B, H, num_steps_past=T, d]，new_cache追加了时间步T
        return y, new_cache

# =============== MLP ===============
# [..., D] --linear-porj--> [..., 4*D] --gelu--> [..., 4D] --linear-proj--> [..., D] --dropout--> [..., D]
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
# kv_cache(if any)-->
# x --layernorm-----> --SelfAttention--> --add_layernorm_ffn_add--> output
#                                    --> new_kv_cache


# 对比 Post-LN from transformer

# kv_cache(if any)-->
# x               --> --SelfAttention--> --add_layernorm--> (--XEncAttention--> add_layernorm) --ffn_add_layernorm--> output
#                                        --> new_kv_cache

# 如果去掉cross-encoder相关, 应该如下

# kv_cache(if any)-->
# x               --> --SelfAttention--> --add_layernorm_ffn_add_layernorm--> output
#                                    --> new_kv_cache
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
        input_token_seqs: torch.Tensor,                 # [B, S]，记录 input_token_seqs 的latest timestep = T
                                                        # train时 S=T+1（0至T），且可以是变长的. 不过同一batch内是确定的. 在infer时 S=1

        attention_mask: Optional[torch.Tensor] = None,  # [B, S], 与 input_token_seqs 保持一致. 1=非PAD, 0=PAD
                                                        # 按理 attention_mask 需要形状 [B, num_steps_so_far=T+1（代表0至T）]
                                                        # train时, 因为S=T+1故输入的attention_mask已经满足
                                                        # infer时, S=1，需要额外为 0<->T-1 时间步追加 全1 的attention mask，寓意past cached 全都是非PAD

        use_cache: bool = False,                        # 若为True, 则返回 [B, H, num_steps_so_far=T+1（时间步0至T）, d] 的 kv-pair cache tuple

        past_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None, 
                                                        # train 时为 None
                                                        # infer时, 首次可以为None（没有past）. 除此之外，它是 tuple of kv_cached pair for blocks
                                                        # 每个(k,v) 是各层的 pair of kv_cached [B, H, num_steps_past=T（时间步0至T-1）, d]
                                                        # infer时由于S=1, 其时间步信息T，可以从这里的dim-2得知
    ):
        B, S = input_token_seqs.shape
        device = input_token_seqs.device

        # 位置索引（GPT-2 使用绝对位置）.
        # train 时 input_token_seqs [B, S=T+1]，token embedding [B, S, D]
        # train 时 position embedding 直接加到 input token embedding上形成 x, 因为train时是self-attention, 后面 qkv 都等于 x
        # 所以 train 时 position embedding 需要的全部位置是 0<->T 所有，即 0至S-1，初始位置为0，长度为 S.

        # infer 时 input_token_seqs [B, S=1]，token embedding [B, S=1, D]
        # infer 时 position embedding 只加在 input token embedding 上，也就是只有 q 上. kv 在从cached导入，以及输出new_kv，都已经包含了位置信息
        # 所以 infer 时 position embedding 需要的全部位置只有 T，从 past_kv 的 dim-2 长度可以得到 T = num_steps_past，初始位置为T，长度为 1=S
        if past_kv is None:
            start_position = 0
        else:
            start_position = past_kv[0][0].size(2) 
        pos_ids = torch.arange(start_position, start_position + S, device=device).unsqueeze(0)  # [1, S]

        # token embedding + position embedding
        tok = self.wte(input_token_seqs)    # [B, S] --linear-proj--> [B, S, D]
        pos = self.wpe(pos_ids)             # [1, S] --linear-proj--> [1, S, D]
        x = self.drop(tok + pos)            # x = token_embedding +(broadcast) position_embedding --> [B, S, D]

        # attention_mask 作为对 input_token_seqs 的描述，形状与 input_token_seqs 保持一致都是 [B, S]
        # 在有 past_kv 的时候，需要补全对 past_kv 的描述(past全部视作非PAD)，从而满足 [B, num_steps_so_far=T+1（代表0至T）] 的att计算硬性要求
        if attention_mask is not None and start_position > 0:
            pad = torch.ones(B, start_position, dtype=attention_mask.dtype, device=device) # [B, num_steps_past=T]
            attention_mask = torch.cat([pad, attention_mask], dim=1)  # [B, num_steps_past+S=T+1]

        # 当 user_cache = True 时，需要记录 attention 层产出的 kv
        new_past = [] if use_cache else None

        # 逐层 x_input[B, S, D] --> x_output[B, S, D]
        # if use_cache，那么逐层产出的 kv 都被 cache 到 new_past 容器
        for i, block in enumerate(self.h):
            kv = None if past_kv is None else past_kv[i]
            # 当输入了 past_kv 时，block attention 内部的 kv是 past_kv，否则是 input x 作自注意力
            x, new_kv = block(x, attention_mask=attention_mask, kv_cache=kv, use_cache=use_cache)
            if use_cache:
                new_past.append(new_kv)

        x = self.ln_f(x) # pre-layernorm的架构中，block 的末尾是没有 layer-norm的（只有add）。所以这里结束所有block后，加一次layernorm
        logits = self.lm_head(x)  # [B,S,D] --> [B,S,n_vocab]

        return logits, (tuple(new_past) if use_cache else None)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,                # [B, S], 其中 latest timestep 为 T
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
    ):
        """
        gpt2 model 的输入(主要关注infer时)
        input_token_seqs: [B, S]. 记 input_token_seqs 的latest timestep = T, 在infer时 S=1
        attention_mask: [B, S] or None. 与 input_token_seqs 保持一致. 1=非PAD, 0=PAD. infer时, S=1
        use_cache: 若为True, 则返回 [B, H, num_steps_so_far=T+1(时间步0至T), d] 的 kv-pair cache tuple; 若为False, 则返回None
        past_kv: infer时, 首次可以为None(没有past). 除此之外，它是 tuple of kv_cached pair for blocks.
                 每个(k,v) 是各层的 pair of kv_cached [B, H, num_steps_past=T(时间步0至T-1), d]

        从 gpt2 model 的 forward 可以看出, infer时生成generate是可以并发的, 即合并成一个batch来生成(并发size=B).
        gpt2 model forward要求Batch内部序列长度一致, 在infer时这不是问题, 因为infer时序列长度都是 1.
        在第一次infer, past_kv = None, input_token_seqs = prompt全部即 input_ids with shape [B, S], 记 latest timestep = T. use_cache设为True
        这样forward返回
        1. logits [B, S, n_vocab]. 其中只需要最后一个时间步的预测 preds [B, n_vocab], 它是关于时间步T+1的token预测分布
        2. tuple of pair kv_cached, 其中 k/v cached 形状为 [B, H, num_steps_so_far=T+1, d], 它是关于时间步0至T的tokens tensor
        这个第一次infer的过程叫做prefill.

        后续infer, past_kv = 前一次循环返回的 tuple of pair kv_cached [B, H, num_steps_so_far=T+1, d], 寓意past时间步是0<->T, 
        input_token_seqs 为前一次循环最后一个时间步的预测 preds [B, n_vocab]得到的结果 tokens [B, 1], 它是关于时间步T+1的tokens, 此时latest timestep = T+1
        这样forward返回
        1. logits [B, 1, n_vocab]. 其中只需要最后一个(也是唯一一个)时间步的预测 preds [B, n_vocab], 它是关于时间步T+2的token预测分布
        2. tuple of pair kv_cached, 其中 k/v cached 形状为 [B, H, num_steps_so_far=T+2, d], 它是关于时间步0至T+1的tokens tensor
        这样这个循环就成立了。这个后续infer的过程叫做decode.
        """
        self.eval()
        B, S = input_ids.shape
        device = input_ids.device

        # prefill
        past_kv = None
        attention_mask = torch.ones(B, S, dtype=torch.long, device=device) # attention_mask 要和 

        for _ in range(max_new_tokens):
            logits, past_kv = self(
                input_token_seqs = input_ids[:, -1:].contiguous() if past_kv is not None else input_ids,
                # attention mask 始终保持与 input_token_seqs 形状相同, 都是 [B, S]. decode 阶段 S=1
                attention_mask = torch.ones(B, 1, device=device, dtype=torch.long) if past_kv is not None else attention_mask,
                use_cache = True, # prefill 后, use_cache = True 使得 kv_cache 更新了 past_kv
                past_kv = past_kv, # prefill 时 past_kv = None
            )
            # logits [B, S, n_vocab], decode 阶段时 S = 1  --取last--> [B, n_vocab], 即是next token的预测分布(raw)

            logits = logits[:, -1, :] / max(temperature, 1e-6)  # 调温. 温度>1就是对logits作"平滑", 温度<1就是对logits作"锐化"
            # 温度>1, 平滑logits, 使得低概率相对概率被拉高，从而模型输出更多样化、更随机、更具创造性. 但也加大了胡言乱语的可能性 --> 越热越活跃
            # 温度<1, 锐化logits，使得低概率相对概率被降低，从而模型输出更确定、重复性高，更保守 --> 越热越冷静
            # 本质是因为softmax在logits的线性伸缩下，分布非线性.

            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k, dim=-1) # 在 logits[B, n_vocab] dim-1 取每行的 top_k, 从大到小 从左至右排列
                # values shape: [B, top_k]
                kth = values[..., -1, None] # values 取最后一列, 然后unsqueeze最后一维度 --> kth shape: [B, 1]

                # torch.where(TF_tensor, input_tensor, other_tensor)
                # 这里TF_tensor/input_tensor/other_tensor 的形状相同. True位置填入 input_tensor 相应元素, False位置填入 other_tensor相应元素
                logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
                # 这里其实是消灭了logits[B, n_vocab]每行的 top_k 之外的所有值: 用 -inf替代, 使得softmax后概率为0.

            probs = F.softmax(logits, dim=-1) # [B, n_vocab], 其中 top_k 之外的概率为 0
            # 以 probs 的行作为 多项分布的 概率分布，从中采样 1 个样本，返回样本的 行index. 其实就是依top_k的概率分布采样
            next_token = torch.multinomial(probs, num_samples=1, replacement=False)  # [B, 1], 作为 next-token的预测
            input_ids = torch.cat([input_ids, next_token], dim=1) # 追加更新到 input_ids 的最后一列.
            # 它将在下一循环以 input_ids[:, -1:]的方式进入模型以推理下一个token

            # 更新attention mask, 追加一列全1到原attention mask, 使之shape与input_ids保持一致.
            # 实际上这个attention mask变量不会再在循环中进入模型. 应该只是一致性操作
            # 此外在实际中, 应该根据生成的token是否是eos, 来确定追加的attention mask列. 如果生成token是eos, 那么mask值应该是0
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

