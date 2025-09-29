# Loss for GPT
# network of gpt2 forward during train:
# INPUT:
#   input_seqs: data of tensor[B, L_q]int64, L_q <= MAX_CONTEXT_SIZE
#       MAX_CONTEXT_SIZE 是模型的参数, 会被用在 casual_mask / position encoding 两个结构部分(可能). 代表最大的 因果遮罩大小 / 可编程位置数量.
#       L_q 是 data batch 的 sequence length(PAD后). 在 单一 batch 内保持固定, 不同 batch 可以不同, 但 L_q 始终小于等于 MAX_CONTEXT_SIZE
#       由于 MAX_CONTEXT_SIZE 是 模型宣称的参数, 意指 该模型能捕捉 最大为该值的上下文关系. 为了达到宣称的效果, 实际训练的 L_q 应该接近 MAX_CONTEXT_SIZE.
#   input_segs: None | tensor[B, L_q]int64
#       当不为 None 时, 形状与 input_seqs 相同. 值代表每个 TOKEN 所属的 TEXT ID. 0 代表 PAD.
#   past_kv: None
#   past_attention_mask: None
#   if_cache_kv: False

# OUTPUT:
# 1. logits: tensor [B, L_q, vocab_size]float
# 2. None
# 3. segments: INPUT 的 input_segs 原样输出.

# LABEL:
# label_seqs: label of tensor[B, L_q]int64
#       label_seqs 与 input_seqs 的对齐方式 alignment:
#           对于 input_seqs 的 PAD      <--------------------->     label_seqs 也是 PAD
#           对于 input_seqs 的 TOKEN    ---原corpus往后移1位--->     label_seqs 的 TOKEN
# 但是在 text-packing 的数据下, 会由 跨文档生成. 

# 考虑 两个text packing: '<PAD><PAD>abc0def0', 0 是 <PAD>/ENDOFTEXT TOKEN. 产出 input_seqs '00abc0def', label_seqs '00bc0def0'. 自回归系数矩阵:
#        0  0  a  b  c  0  d  e  f
#     0  x                              --> 0 (mask because PAD)
#     0  x  x                           --> 0 (mask because PAD) / a (mask because q&label not same text)
#     a  x  x  1                        --> b
#     b  x  x  1  1                     --> c
#     c  x  x  1  1  1                  --> 0
#     0  x  x  1  1  1  1               --> d (mask because q&label not same text)
#     d  x  x  /  /  /  /  1            --> e
#     e  x  x  /  /  /  /  1  1         --> f
#     f  x  x  /  /  /  /  1  1  1      --> 0

# 有两个地方可能会存在跨文档: 如果 label seq 写作 0abc0def0 (这样就是原 whole seq 的 1 到 末尾, 比较容易生产), 那么
#   q=0(PAD) --> label=a  和 q=0(ENDOFTEXT1) --> label=d 这两个地方是跨文档的, 应该 mask 屏蔽掉.
# label_mask 的生成逻辑: 当且只有当 input_seg == label_seg != 0(PAD), 即"同一个TEXT文档且都不为PAD" --> True, otherwise --> False


import torch.nn as nn
from ...core.loss.mask_ce_loss import MaskedCrossEntropyLoss
from torch import Tensor

class gpt2_pretrain_loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = MaskedCrossEntropyLoss(reduction='none')
    
    def forward(self, logits: Tensor, input_segs: None|Tensor, labels: Tensor, label_segs: None|Tensor) -> Tensor:
        # logits: [B, L_q, vocab_size]float
        # labels: [B, L_q]int64
        # input_segs: None | tensor[B, L_q]int64
        # label_segs: None | tensor[B, L_q]int64
        if input_segs is not None and label_segs is not None:
            label_mask = (input_segs == label_segs) * (label_segs != 0) # [B, L_q]
        else:
            label_mask = None
        
        return self.loss(logits.transpose(1, 2), labels, label_mask) #[B, L_q]