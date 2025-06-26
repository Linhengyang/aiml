# test.py
import typing as t

def get_pair_counts(tokens:t.List[int], p_counts:t.Dict[tuple[int, int], int]|None = None):
    '''
    p_counts 如果是 None: 那么就是 给定 tokens list, 计算它的 pair-tokens counts, 返回一个 pair counts dict
    p_counts 如果不是 None: 那么就是 给定 tokens list, 计算它的 pair-tokens counts, 更新 到 输入的 p_counts 里
    '''
    if not p_counts:
        p_counts = {}
    for pair in zip(tokens, tokens[1:]):
        p_counts[pair] = p_counts.get(pair, 0) + 1

    return p_counts


if __name__ == "__main__":
    tokens = []
    print(get_pair_counts(tokens))

    p_counts = {}
    returned = get_pair_counts(tokens, p_counts)

    print(returned)

    
