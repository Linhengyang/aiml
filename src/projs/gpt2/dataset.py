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
from ...core.utils.text.tokenizer import boostBBPETokenizer, ENDOFTEXT
from ...core.utils.common.seq_operation import pack_seq_to_batch_slide



class mtDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len, overlap=0, pad_value=0):
        super().__init__()
        data = torch.load(data_path)

        input_seq_seg = pack_seq_to_batch_slide(data[:, :-1], seq_len, overlap, pad_value) # [B, 2, seq_len]
        label_seq_seg = pack_seq_to_batch_slide(data[:, 1:], seq_len, overlap, pad_value) # [B, 2, seq_len]

        self.input_seqs = input_seq_seg[:, 0, :] # [B, seq_len]
        self.input_segs = input_seq_seg[:, 1, :] # [B, seq_len]

        self.labels = label_seq_seg[:, 0, :] # [B, seq_len]
        self.label_segs = label_seq_seg[:, 1, :] # [B, seq_len]

        self._size = self.input_seqs.size(0)

    def __getitem__(self, index):
        return self.input_seqs[index], self.input_segs[index], self.labels[index], self.label_segs[index]
    
    def __len__(self):
        return self._size
    