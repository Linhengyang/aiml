# Loss for GPT
# network of gpt2 forward during train:
# INPUT:
#   input_seqs: data of tensor[B, L_q]int64, L_q <= MAX_CONTEXT_SIZE
#       MAX_CONTEXT_SIZE 是模型的参数, 会被用在 casual_mask / position encoding 两个结构部分(可能). 代表最大的 因果遮罩大小 / 可编程位置数量.
#       L_q 是 data batch 的 sequence length(PAD后). 在 单一 batch 内保持固定, 不同 batch 可以不同, 但 L_q 始终小于等于 MAX_CONTEXT_SIZE
#       由于 MAX_CONTEXT_SIZE 是 模型宣称的参数, 意指 该模型能捕捉 最大为该值的上下文关系. 为了达到宣称的效果, 实际训练的 L_q 应该接近 MAX_CONTEXT_SIZE.
#   attention_mask: None 仅当 input_seqs 全部都不是 PAD. / tensor [B, L_q]bool, 对 input_seqs 的 PAD-info 描述. False-->PAD, True-->non-PAD.
#       PAD 和 EODOFTEXT 分隔符等特殊作用符不是一个概念. 特殊作用符号是valid token, 即它们正常参与模型计算和参数更新. 
#       PAD 则不同, 它们会被 attention_mask 从运算中彻底屏蔽掉, 不参与模型运算. PAD 仅仅只是为了补齐 batch 内 部分序列样本 到一致长度. 尽量减少PAD.
#       因为 PAD 会在运算中被彻底屏蔽, 所以用 ENDOFTEXT等特殊作用符 来当作 PAD 只是为了复用符号, 并不是说 PAD 位置 是 TOKEN.
#   past_kv: None
#   past_attention_mask: None
#   if_cache_kv: False

# OUTPUT:
# 1. logits: tensor [B, L_q, vocab_size]float
# 2. None
# 3. attention_mask: INPUT 的 attention_mask 原样输出

# LABEL:
# label_seqs: label of tensor[B, L_q]int64
#       label_seqs 与 input_seqs 的对齐 align:
#           对于 input_seqs 的 PAD      <----------------=---->     label_seqs 也是 PAD
#           对于 input_seqs 的 TOKEN    ---原corpus往后移1位--->     label_seqs 的 TOKEN
# 所以可以看出, attention_mask 同时也是 label_seqs 的 PAD-info. 所以它直接参与 构造 label_mask.
# label_mask 还需要其他一些信息, 以继续屏蔽一些 sequence position. 比如 text-packing seqs 里, label 里每个 text 的 first-token 不计loss

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

# attention_mask = nonpad_mask ^ q&k_sametext_mask, attention 内部还要 ^ casual_mask
# label_maks = notpad ^ q&label_sametext. 而实际上, 若视 PAD 和 TOKEN 总是 not same text, label_mask 只需一条准则: q&label same text or both PAD
# 这里 ^ 是 且运算. 


import torch.nn as nn
import torch


class gpt2loss(nn.Module):
    pass