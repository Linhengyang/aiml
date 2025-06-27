import typing as t


def truncate_pad(line, num_steps, padding_token):
    '''
    inputs: line, num_steps, padding_token
        text: 1D list
        num_steps: integer, setup sequence length
        padding_token: element to pad up to num_steps length

    returns: a 1D list, denoted as L

    explains:
        truncate what exceeds num_steps, or pad padding_token when it shorts
    '''
    if len(line) >= num_steps:
        return line[:num_steps]
    else:
        return line + [padding_token] * (num_steps - len(line))



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