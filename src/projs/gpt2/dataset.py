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
#    可以看出, 若 ENDOFTEXT token 进入 input_seqs, 会混淆前后两篇不相干的文章, 会干扰模型的上下文, 干扰模型的学习能力.
#    所以一般不把 ENDOFTEXT token 做入 input_seqs. ???


# 4. corpus 切分成 input_seqs 时, 是否考虑 overlapping? 考虑对于 corpus 'abcdef0', 0代表 ENDOFTEXT, L_q = 3, 产出 data-label pair
#       方案一 no-overlapping: (abc, bcd), (def, ef0)
#       方案二 overlap 1, stride = 2: (abc, bcd), (cde, def), (ef<PAD>, f0<PAD>)
#       方案三 overlap 2, stride = 1: (abc, bcd), (bcd, cde), (cde, def), (def, ef0), (ef<PAD>, f0<PAD>), (f<PAD><PAD>, 0<PAD><PAD>)
#    首选方案一: 样本相关性最低, 并与 packing+分段因果遮罩 天然相容. 当小样本微调时, 可以考虑方案二: overlap要小/stride要大, 尽量不PAD(packing或丢弃)


import torch

class projDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        pass
    
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass
    