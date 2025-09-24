# dataset for GPT
# 如何 从 corpus 生产 input_seqs 和 label_seqs?
# 1. 确定 L_q. L_q 小于等于 MAX_CONTEXT_SIZE, 最好接近它以保证宣称的效果. 不同 batch 的 L_q 可以不同, 有些模型在 epoch 加大时, 逐渐增大 L_q
# 2. corpus 各 text 末尾都要 append ENDOFTEXT 符, 以保证 ENDOFTEXT TOKEN 以正确的方式进入 label_seqs, 从而让模型正确学习何时该生成 ENDOFTEXT
# 3. text 是否考虑 packing 成一条序列? 作 ENDOFTEXT 进入 input_seqs 影响分析:
#    考虑 corpus 'abcd0', 0 是 ENDOFTEXT TOKEN. 产出 input_seqs 'abcd', label_seqs 'bcd0'. 可见 ENDOFTEXT 不进入 input_seq时, 对齐是自然的
#    考虑 两个text packing: 'abc0def0', 0 是 ENDOFTEXT TOKEN. 产出 input_seqs 'abc0def', label_seqs 'bc0def0'. 我们列出 自回归系数矩阵:
#          a  b  c  0  d  e  f
#       a  1                     --> b
#       b  1  1                  --> c
#       c  1  1  1               --> 0
#       0  1  1  1  1            --> d
#       d  1  1  1  1  1         --> e
#       e  1  1  1  1  1  1      --> f
#       f  1  1  1  1  1  1  1   --> 0
#    从上往下数, 一直到 c --> 0 都是正常的. 但是下一行 0 --> d 就值得商榷了: ENDOFTEXT 作为 q, 前文 abc0 作为 kv, label 是下一篇文章的开头 d?
#    再往下看, d --> e: d 作为 q, ENDOFTEXT 和前一篇文章的 abc 作为 kv, label是本文的 e?
#    可以看出, 若 ENDOFTEXT token 进入 input_seqs, 会造成两个问题: 1. 混淆前后 不相干的文章 2. label为文档首token时, 对应q为 ENDOFTEXT, 这是错误的.

#    但实际上, ENDOFTEXT token 进入 input_seqs 即 text-packing 成一条sequence 是刚需, 因为模型需要学会处理 ENDOFTEXT, ENDOFTEXT 需要在 input_seqs
#    中出现, 这样才能使得 ENDOFTEXT 的 embedding 以及相关表示权重得到更新.(ENDOFTEXT 若只作为label出现, 是能学习预测生成它，而没办法学习表征它)

#    如何解决 不相干的文档混淆问题？---> text-packing + block-diagonal attention_mask, 即 attention_mask 除了屏蔽PAD之外, 还要屏蔽非同文档TOKEN
#       即 x 区域要屏蔽
#          a  b  c  0  d  e  f
#       a  1                     --> b
#       b  1  1                  --> c
#       c  1  1  1               --> 0
#       0  1  1  1  1            --> d
#       d  x  x  x  x  1         --> e
#       e  x  x  x  x  1  1      --> f
#       f  x  x  x  x  1  1  1   --> 0

#    如何解决 EODOFTEXT 作为 q 生成 下一文档首TOKEN 的对齐问题 ---> valid-prev-mapping label_mask, 即 label_mask 除了屏蔽PAD之外, 还要屏蔽非法前驱TOKEN
#       判断一个 label 是否合法，当且仅当以下两种情况: 1. q 为 PAD, label 也为 PAD；2. q 是 label  的同一文档的左邻TOKEN
#       故上述 7 对 q-label pair 中, a->b / b->c / c->0 , d->e / e->f / f-> 0 都是合法的, 而 0->d 是非法的, 因为这里 0 和 d是不同文档的TOKEN.
#       得 label_mask = [1,1,1,0,1,1,1]. 注意该法则在实质上已经包含了 PAD-info(PAD->token / token -> PAD 都是非法)


# 4. corpus 切分成 input_seqs 时, 是否考虑 overlapping? 考虑对于 corpus 'abcdef0', 0代表 ENDOFTEXT, L_q = 3, 产出 data-label pair
#       方案一 no-overlapping: (abc, bcd), (def, ef0)
#       方案二 overlap 1, stride = 2: (abc, bcd), (cde, def), (ef<PAD>, f0<PAD>)
#       方案三 overlap 2, stride = 1: (abc, bcd), (bcd, cde), (cde, def), (def, ef0), (ef<PAD>, f0<PAD>), (f<PAD><PAD>, 0<PAD><PAD>)
#    首选方案一: 样本相关性最低, 并与 packing+分段因果遮罩 天然相容. 当小样本微调时, 可以考虑方案二: overlap要小/stride要大, 尽量不PAD(packing或丢弃)


import torch
import typing as t
import math
from ...core.utils.text.tokenizer import boostBBPETokenizer
from dataclasses import dataclass

# 用一个三维 torch.Tensor 代表一批同长序列: [B, 3, L] ---> seqBox
# 指 B 个 datapoint, 每个 datapoint 是 三个长度为 L 的序列, 分别为 input 序列 / segments 序列 / label 序列

# data 是以 seqBox 的长度为 key, seqBox 为 value 的字典


def _regularize_batch(S: int, P: int, data: t.Dict[int, torch.Tensor]):
    '''
    把长度从 S(包含) 到 P(不包含) 的所有 seqBox, 都拆散. 其中整块的 S 部分并入 长度为S的 seqBox, 剩余零碎的 b 部分并入相应 长度为b的 seqBox
    '''
    # 对于 L = S 的 seq_box, 它是现成的, 无需操作 --> [_, 3, S]
    if S not in data:
        data[S] = torch.empty(0, 3, S, dtype=torch.long) # 空 Tensor, [0, 3, S]
    
    output = data[S] # 不涉及copy. torch 的好处就是这里的行为统一引用复用地址

    # 对于 L > S 的 seq_box, L = kS + b, k >=1, 0 <= b < S
    for L in range(S+1, P):
        if L in data: # data[L]: [B, 3, L]
            k, b = L // S, L % S
            splitted = data[L].split(S, dim=-1) # [B, 3, kS+b] --split--> k 个 [B, 3, S], 1 个 [B, 3, b]
            if b == 0:
                output = torch.concat([output, *splitted], dim=0) # [_, 3, S] stack k 个 [B, 3, S]
            else:
                output = torch.concat([output, *splitted[:-1]], dim=0) # [_, 3, S] stack k 个 [B, 3, S]
                # [_, 3, b] --> [_+B, 3, b]
                data[b] = torch.concat([data.get(b, torch.empty(0, 3, b, dtype=torch.long)), splitted[-1]], dim=0)
            del data[L] # L 已经拆散分配到 S 和 b. 分配完毕后 从 data 中消除对应 kv


def _size_stat(S: int, P: int, data: t.Dict[int, torch.Tensor]):
    '''
    计算长度从 S(包含) 到 P(不包含) 的 所有 seqBox 的总 size
    '''
    _size = 0
    for L in range(S+1, P):
        if L in data:
            _size += data[L].size(0) # [B, 3, L]
    return _size


def _pad(data: torch.Tensor, L):
    # data shape: [B, 3, q]
    # pad to [B, 3, 0]
    q = data.size(-1)
    if q < L:
        zeros = torch.zeros(data.shape[0], data.shape[1], L-q)
        return torch.cat([data, zeros], dim=-1)
    elif q > L:
        return data[:, :, :L]
    else:
        return data


def pack_text(tgt_L: int, data: t.Dict[int, torch.Tensor]):
    '''
    对于长度 大于等于 tgt_L 的 seqBox, regularize 到 tgt_L. 计算此时 tgt_L/2 到 tgt_L 的seqBox的总size B, 以及1到tgt_L的总size B_
    如果 B >= 2, 则执行下一步; 如果 B < 2 但是 B_ >=2, 则把所有 1到tgt_L/2 的都PAD到 tgt_L/2, 然后执行下一步后结束; 如果 B_ < 2, 结束

    对于长度 大于等于 tgt_L/2 但又小于 tgt_L 的seqBox, regularize 到 tgt_L/2 后 得到 [B, 3, tgt_L/2] 的 seqBox.
    拆分 dim0 pack 到 dim2, 得到 [B//2, 3, tgt_L]. 这里可能要pack 1条 [1, 3, tgt_L/2] 的 datapoint 到 residual

    对于长度 大于等于 tgt_L/4 但又小于 tgt_L/2 的seqBox, regularize 到 tgt_L/4 后 得到 [B, 3, tgt_L/4] 的 seqBox.
    拆分 dim0 pack 到 dim2, 得到 [B//4, 3, tgt_L]. 这里可能要pack 3条 [1, 3, tgt_L/4] 的 datapoint 到 residual.

    ...

    重复上述流程 t-1 次后, 检查所有长度小于 tgt_L/2^t 的 seqBoxes, 计算 min_L, 以及总size B. 若 min_L < tgt_L/2^t, B >= 2^t 成立,
    步骤一: 对于长度小于 tgt_L/2^t 的, 全部 PAD 到 长度 tgt_L/2^t.
    步骤二: 对于长度 大于等于 tgt_L/2^t 但又小于 tgt_L/2^(t-1) 的seqBox, regularize 到 tgt_L/2^t 后 得到 [B, 3, tgt_L/2^t] 的 seqBox.
    拆分 dim0 pack 到 dim2, 得到 [B//2^t, 3, tgt_L]. 这里可能要 pack 2^t-1 条 [1, 3, tgt_L/2^t] 的 datapoints 到 residual.
    '''
    min_L, max_L = min(data), max(data)
    T = int( math.log2(tgt_L/min_L) )
    
    _regularize_batch(tgt_L, max_L+1, data) # what if tgt_L >= max_L?
    # 此时 data: min_L -- tgt_L

    residual = torch.empty(1, 3, 0)

    for t in range(1, T+1): # t 最多到 T, 因为当 t > T 时, min_L > tgt_L/2^t, 而裁剪 min_L 实属没有必要.
        # 考虑子区间 tgt_L/2^t -- tgt_L/2^(t-1), 即:
        # t = 1: tgt_L/2 -- tgt_L
        # t = 2: tgt_L/4 -- tgt_L/2
        # ...
        # t = t: tgt_L/2^t -- tgt_L/2^(t-1)
        L_t = tgt_L//2**t
        up_semi_size = _size_stat(L_t, tgt_L//2**(t-1), data) # size from tgt_L/2^t(incl) -- tgt_L/2^(t-1)(not-incl)
        down_semi_size = _size_stat(min_L, L_t, data) # size from min_L(incl) -- tgt_L/2^t(not-incl)

        if up_semi_size >= 2**t:
            _regularize_batch(L_t, tgt_L//2**(t-1))
            temp = data[L_t] # [up_semi_size, 3, tgt_L//2**t]
            # up_semi_size = 2**t * up_semi_size//2**t + r, 0 <= r < 2**t
            incre_size, r = up_semi_size//(2**t), up_semi_size%(2**t)
            splitted = temp.split(incre_size, dim=0) # 2**t个 [incre_size, 3, tgt_L//2**t], 1个 [r, 3, tgt_L//2**t]
            if r != 0:
                increment = _pad( torch.cat(splitted[:-1], dim=-1), tgt_L ) # [incre_size, 3, tgt_L]
                # 补充 last [r, 3, tgt_L//2**t] 到 [2**t, 3, tgt_L//2**t]
                residual = torch.cat([splitted[-1], torch.zeros(2**t-r, 3, L_t, dtype=torch.long)], dim=0)
                residual = _pad( torch.cat(residual.split(1, dim=0), dim=-1), tgt_L ) # [2**t, 3, tgt_L//2**t] --> [1, 3, tgt_L]
                increment = torch.cat([increment, residual], dim=0) # [.., 3, tgt_L]
            else:
                increment = _pad( torch.cat(splitted, dim=-1), tgt_L ) # [.., 3, tgt_L]
            data[tgt_L] = torch.cat([data[tgt_L], increment], dim=0)
        elif up_semi_size + down_semi_size >= 2**t:
            _size = up_semi_size + down_semi_size
            # 遍历所有 长度小于 tgt_L//2**t 的 seqBox, 将它们的长度全部 PAD 到 tgt_L//2**t, 然后全部 stack 到 长度为 tgt_L//2**t 的 seqBox 上
            if L_t not in data:
                data[L_t] = torch.empty(0, 3, L_t, dtype=torch.long) # [.., 3, tgt_L//2**t]
            
            for l in range(min_L, L_t):
                if l in data:
                    data[L_t] = torch.cat([data[L_t], _pad(data[l], L_t)], dim=0) # [..++, 3, tgt_L//2**t]
                    del data[l]
            _regularize_batch(L_t, tgt_L//2**(t-1))
            temp = data[L_t] # [_size, 3, tgt_L//2**t]
            # _size = 2**t * incre_size + r, 0 <= r < 2**t
            incre_size, r = _size//(2**t), _size%(2**t)
            splitted = temp.split(incre_size, dim=0) # 2**t个 [incre_size, 3, tgt_L//2**t], 1个 [r, 3, tgt_L//2**t]
            if r != 0:
                increment = _pad( torch.cat(splitted[:-1], dim=-1), tgt_L ) # [incre_size, 3, tgt_L]
                # 补充 last [r, 3, tgt_L//2**t] 到 [2**t, 3, tgt_L//2**t]
                residual = torch.cat([splitted[-1], torch.zeros(2**t-r, 3, L_t, dtype=torch.long)], dim=0)
                residual = _pad( torch.cat(residual.split(1, dim=0), dim=-1), tgt_L ) # [2**t, 3, tgt_L//2**t] --> [1, 3, tgt_L]
                increment = torch.cat([increment, residual], dim=0) # [.., 3, tgt_L]
            else:
                increment = _pad( torch.cat(splitted, dim=-1), tgt_L ) # [.., 3, tgt_L]
            data[tgt_L] = torch.cat([data[tgt_L], increment], dim=0)
        else:
            break

    return data[tgt_L]




class projDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        pass
    
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass
    