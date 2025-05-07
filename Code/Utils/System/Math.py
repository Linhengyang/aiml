# Math.py
import math




def find_closet_2factors(N):

    # 从 N 的平方根开始向下搜索. N 即使是 质数也会得到 (1, N) 的正确结果
    start = int(math.sqrt(N))

    for x in range(start, 0, -1):
        if N % x == 0:
            y = N // x
            return x, y



