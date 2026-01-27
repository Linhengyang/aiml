import torch

def next_token_topk(
        logits: torch.Tensor,       # [B, vocab_size]
        temperature: float,
        top_k: int|None
        ):
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