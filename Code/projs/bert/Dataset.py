import torch

def get_tokens_segments(tokens_a, tokens_b=None):
    # input token_a: list of tokens, [tk1, tk2,...,tkN]
    # input token_b if not none: list of tokens, [tk1, tk2,...,tkN]
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * len(tokens)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1]*(len(tokens_b) + 1)
    return tokens, segments