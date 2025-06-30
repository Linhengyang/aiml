# Math.py
import math
import numpy as np



def find_closet_2factors(N:int) -> tuple[int, int]:
    assert isinstance(N, int) and N > 1, f"N must be a integer larger than 1"
    
    # 从 N 的平方根开始向下搜索. N 即使是 质数也会得到 (1, N) 的正确结果
    start = int(math.sqrt(N))

    for x in range(start, 0, -1):
        if N % x == 0:
            y = N // x
            return x, y



def cosine_similarity(a:np.ndarray, b:np.ndarray) -> np.float64:
    assert a.ndim == b.ndim == 1, f'input arrays must be 1D array'

    # a, b: shape (L, )
    dot_prod = np.dot(a, b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    cosine_sim = dot_prod / (norm_a*norm_b)

    return cosine_sim