import torch
import torch.nn as nn
import typing as t
import math

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
    




def _regularize_batch(S: int, E: int, data: t.Dict[int, torch.Tensor], n: int):
    '''
    把长度 L 从 S(包含) 到 E(不包含) 的所有 [B, n, L] 拆散. 其中整块的 S 部分并入 长度为 S 的 seqBox, 剩余零碎的 余数长度 部分并入相应长度的 seqBox
    '''
    # 对于 L = S 的 seq_box, 它是现成的, 无需操作 --> [_, n, S]
    if S not in data:
        data[S] = torch.empty(0, n, S, dtype=torch.long) # 空 Tensor, [0, n, S]

    # 对于 L > S 的 seq_box, L = kS + b, k >=1, 0 <= b < S
    for L in range(S+1, E):
        if L in data: # data[L]: [B, n, L]
            k, b = L // S, L % S
            splitted = data[L].split(S, dim=-1) # [B, n, kS+b] --split--> k 个 [B, n, S], 1 个 [B, n, b]
            if b == 0:
                data[S] = torch.concat([data[S], *splitted], dim=0) # concat [_, n, S] 和 k 个 [B, n, S]
            else:
                data[S] = torch.concat([data[S], *splitted[:-1]], dim=0) # concat [_, n, S] 和 k 个 [B, n, S]
                # 余数长度 b 部分 concat 到相应 seqBox: [_, n, b] --> [_+B, 3, b]
                data[b] = torch.concat([data.get(b, torch.empty(0, n, b, dtype=torch.long)), splitted[-1]], dim=0)
            del data[L] # L 已经拆散分配到 S 和 b. 分配完毕后 从 data 中消除对应 kv


def _size_stat(S: int, E: int, data: t.Dict[int, torch.Tensor]):
    '''
    计算长度从 S(包含) 到 E(不包含) 的 所有 seqBox 的总 size
    '''
    _size = 0
    for L in range(S, E):
        if L in data:
            _size += data[L].size(0) # [B, n, L], B 累加
    return _size


def _pad(t: torch.Tensor, tgt_L: int, pad_dim: int = -1, pad_value: int = 0):
    # t shape: [B, n, q]
    # if pad_dim = -1, --> pad to [B, n, L]; if pad_dim = 0, --> pad to [L, n, q]
    q = t.size(pad_dim)
    if q < tgt_L:
        pad_shape = list(t.shape)
        pad_shape[pad_dim] = tgt_L - q
        pads = torch.full(pad_shape, fill_value = pad_value, dtype=torch.long)
        return torch.cat([t, pads], dim=pad_dim)
    else:
        return t


def pack_seq_to_batch_pow2(data: t.Dict[int, torch.Tensor], tgt_L: int, min_L: int, pad_value: int = 0):
    '''
    data: 字典 key: L, 等于序列长度, value: tensor[B, n, L], 表示 B 条长度为 L 的序列, 每个序列包括 input/label/segments/... 共 n 种
          data 按 key 从小到大排序 ----------> {1:[B1, n, 1], 2:[B2, n, 2], ..., L:[B, n, L],...}
    tgt_L: int, 表示最终 packed batch 中的序列长度. 大于 tgt_L 的部分会被不断 以2的幂次的比例 切割, 然后跨文档 pack 到 tgt_L 的长度.
    min_L: int, 表示在 packing 过程中, 小于 min_L 的序列会被丢弃.
    '''
    T = int( math.log2(tgt_L/min_L) )
    assert T >= 0 # T = 0 是 OK 的, 执行完第一次 regularize to batch 之后, 不再执行后续循环

    max_L = max(data) # 本方法不能解决 tgt_L < max_L 的情形.
    assert tgt_L < max_L, f'target batch sequence length must be smaller than max sequence length of data'
    n = data[max_L].size(1) # {1:[B1, n, 1], 2:[B2, n, 2], ..., L:[B, n, L],...}

    _regularize_batch(tgt_L, max_L+1, data, n)

    residual = torch.empty(1, n, 0, dtype=torch.long) # 零碎的 residual 会 concat 到 dim-1

    for t in range(1, T+1): # t 最多到 T, 因为当 t > T 时, min_L > tgt_L/2^t, min_L 以下丢弃
        pow2_t, pow2_t_ = 2**t, 2**(t-1)
        # 关注区间 min_L <--> tgt_L//2^t ,  tgt_L//2^t <-->  tgt_L//2^(t-1)
        mid_L, rend_L = tgt_L//pow2_t, tgt_L//pow2_t_
        upper_B = _size_stat(mid_L, rend_L, data) # mid_L(含) 到 rend_L(不含) 的 总B size
        lower_B = _size_stat(min_L, mid_L, data) # min_L(含) 到 mid_L(不含) 的 总B size

        if upper_B >= pow2_t: # 当 右半区间 总size即可满足 pack to tgt_L
            _alloc_size = upper_B
        elif upper_B + lower_B >= pow2_t: # 当 左右半区间 加起来 总size 可满足 pack to tgt_L
            _alloc_size = upper_B + lower_B
            # 遍历所有 长度小于 mid_L 的 seqBox, 将它们的长度全部 PAD 到 mid_L, 然后全部 concat 到 长度为 mid_L 上
            if mid_L not in data:
                data[mid_L] = torch.empty(0, n, mid_L, dtype=torch.long)
            
            for l in range(min_L, mid_L):
                if l in data:
                    pads = _pad(data[l], mid_L, -1, pad_value) # [_, n, l] --> [_, n, mid_L]
                    data[mid_L] = torch.cat([data[mid_L], pads], dim=0) # [_+_, n, mid_L]
                    del data[l]
        else:
            return data[tgt_L], data
        
        incre_B, rsd = _alloc_size//pow2_t, _alloc_size%pow2_t # _alloc_size = pow2_t*incre_B, + rsd, 0 <= rsd < pow2_t
        _regularize_batch(mid_L, rend_L, data, n)

        splitted = data[mid_L].split(incre_B, dim=0) # pow2_t个 [incre_B, n, mid_L], 1个 [rsd, n, mid_L]
        if rsd != 0:
            increment = _pad(torch.cat(splitted[:-1], dim=-1), tgt_L, -1, pad_value) # [incre_B, n, tgt_L]
            residual = _pad(splitted[-1], pow2_t, 0, pad_value) # PAD residual [rsd, n, mid_L] -> [pow2_t, n, mid_L]
            residual = _pad(torch.cat(residual.split(1, dim=0), dim=-1), tgt_L, -1, pad_value) # [pow2_t, n, mid_L] --> [1, n, tgt_L]
            increment = torch.cat([increment, residual], dim=0) # [incre_B+1, n, tgt_L]
        else:
            increment = _pad(torch.cat(splitted[:-1], dim=-1), tgt_L, -1, pad_value) # [incre_B, n, tgt_L]
        data[tgt_L] = torch.cat([data[tgt_L], increment], dim=0)

        del data[mid_L]

    return data[tgt_L], data




def pack_seq_to_batch_slide(seq: torch.Tensor, tgt_L: int, overlap: int = 0, pad_value: int = 0):
    '''
    seq: tensor of [n, cat_L] 或 [cat_L,]  即 n或1 条 长度为 cat_L 的序列. cat_L 是指该序列可能是 concated sequence
    tgt_L: 目标 seq batch [B, n, tgt_L] 的长度. 从 cat_L 上以 tgt_L-overlap 为步距, tgt_L 为滑动窗口, 滑动得到 batch
    overlap: 在 cat_L 上滑动时的重叠长度, tgt_L - overlap = 滑动步距. 默认为 0 即默认前后滑动窗口没有重叠
    pad_value: 
    '''
    assert overlap < tgt_L, f'overlap must smaller than tgt_L'
    stride = tgt_L - overlap

    if len(seq.shape) == 2:
        cat_L  = seq.size(1)
        assert cat_L >= tgt_L
        # pad: (last dim 左，last dim 右，last 2nd dim 上，last 2nd dim 下...)
        seq = nn.functional.pad(seq, (0, stride - (cat_L-tgt_L)%stride), mode='constant', value=pad_value)
        # [n, L]/[L, 1] --unfold_on_1--> [n, L/w, w]/[L, stride, 1] --transpose_01--> [L/w, n, w]/[stride, L, 1]
        return seq.unfold(-1, tgt_L, stride).transpose(0,1) # [L/w=B, n, w]/[stride, L, 1]
    elif len(seq.shape) == 1:
        cat_L  = seq.size(0)
        assert cat_L >= tgt_L
        seq = nn.functional.pad(seq, (0, stride - (cat_L-tgt_L)%stride), mode='constant', value=pad_value)
        # [L,]/[1,] --unfold_on_0--> [L/w, w]/[stride, 1]
        return seq.unfold(-1, tgt_L, stride)
    else:
        raise ValueError(f'wrong shape for seq. must be 2D or 1D tensor')