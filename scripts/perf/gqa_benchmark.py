import torch
import time
import math
import torch.nn.functional as F

def benchmark():
    B, H_q, H_kv, L, d = 4, 32, 8, 2048, 128
    n_rep = H_q // H_kv

    q = torch.randn(B, H_q, L, d, device=torch.device('cuda'), dtype=torch.float16)
    k = torch.randn(B, H_kv, L, d, device=torch.device('cuda'), dtype=torch.float16)
    v = torch.randn(B, H_kv, L, d, device=torch.device('cuda'), dtype=torch.float16)

    def method_q_group(q, k, v):
        # (B, H_q, L, d)/(B, H_kv, L, d)/(B, H_kv, L, d)  -->  (B, H_q, L, d)
        scores = torch.matmul(q.view(B, H_kv, n_rep, L, d), k.unsqueeze(2).transpose(-1, -2))/(d**0.5) # B, H_kv, n_rep, L, L
        attn_w = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_w, v.unsqueeze(2)) # (B, H_kv, n_rep, L, L) @ (B, H_kv, 1, L, d)
        return out.view(B, H_q, L, d)
    
    def method_sdpa(q, k, v):
        # (B, H_q, L, d)/(B, H_kv, L, d)/(B, H_kv, L, d)  -->  (B, H_q, L, d)
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
    
    # 预热
    for _ in range(10):
        method_q_group(q, k, v)
        method_sdpa(q, k, v)
    

    # 测试 q-group
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            out1 = method_q_group(q, k, v)
    torch.cuda.synchronize()
    t1 = time.time()

    # 测试 sdpq
    torch.cuda.synchronize()
    t2 = time.time()
    with torch.no_grad():
        for _ in range(100):
            out2 = method_sdpa(q, k, v)
    torch.cuda.synchronize()
    t3 = time.time()

    print(f'q-grouping time: {t1-t0:.4f}s')
    print(f'sdpa time: {t3-t2:.4f}s')
    print(f'diff max: {(out1-out2).abs().max().item()}')





if __name__ == '__main__':
    benchmark()
    
    # fp16 和 bf16 精度下, sdpa 完胜 q-grouping 方案
    # fp32 精度下, q-grouping 方案小胜 sdpa
    # sdpa 无法加入自定义的 attn_mask. 所以归纳如下:
    # q-grouping: fp32精度, 又或者需要自定义 attn_mask
    # sdpa: fp16/bf16精度, 且不需要自定义 attn_mask