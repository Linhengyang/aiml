# math.py
import math
import numpy as np
import typing as t



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




def check_monotonic(
        seq: list,
        mode:t.Literal['increase', 'decrease'],
        strict:bool=True
        ) -> bool:

    if mode == 'increase' and strict:
        for i in range(len(seq) - 1): # 0 至 len(seq) - 2
            if seq[i] >= seq[i+1]: # 假如 存在 当前大于或等于 下一个，即破坏了 严格单调递增
                return False
        return True # 遍历 第 0 至 len(seq) - 2 个元素 与其后一个的比较，若都是 <，说明它是严格单调递增的
    elif mode == 'increase':
        for i in range(len(seq) - 1): # 0 至 len(seq) - 2
            if seq[i] > seq[i+1]: # 假如 存在 当前大于 下一个，即破坏了 单调递增
                return False
        return True # 遍历 第 0 至 len(seq) - 2 个元素 与其后一个的比较，若都是 <=，说明它是单调递增的
    elif mode == 'decrease' and strict:
        for i in range(len(seq) - 1): # 0 至 len(seq) - 2
            if seq[i] <= seq[i+1]: # 假如 存在 当前小于等于 下一个，即破坏了 严格单调递减
                return False
        return True # 遍历 第 0 至 len(seq) - 2 个元素 与其后一个的比较，若都是 >，说明它是严格单调递减的
    elif mode == 'decrease':
        for i in range(len(seq) - 1): # 0 至 len(seq) - 2
            if seq[i] < seq[i+1]: # 假如 存在 当前小于下一个，即破坏了 单调递减
                return False
        return True # 遍历 第 0 至 len(seq) - 2 个元素 与其后一个的比较，若都是 >=，说明它是单调递减的
    else:
        raise ValueError(f'wrong mode for {mode}. must be increase/decrease')
    



def get_compression_ratio(string: str, indices: list[int]) -> float:
    """
    给定 string 和 对应的 tokenized integers 列表，计算 压缩率 = string utf-8字节数量 / tokens 数量
    举例1:
        全英文字符 string, unicode number as token ID, 这样 len(string) = utf-8字节数量 = unicode number 数量
        故 压缩率 = 1
    举例2:
        全中文字符 string, unicode number as token ID, 这样 len(string) = utf-8字节数量 / 3 = unicode number 数量
        故 压缩率 = 3
    """
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    
    return num_bytes / num_tokens