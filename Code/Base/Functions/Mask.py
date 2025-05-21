# Mask.py
import typing as t
import torch





def mask_on_last_dim(tensor_shape, mask_lens, mask_flag=True):
    '''
    input:
        tensor_shape: (..., d_size)
        mask_lens: tensor with shape as (...,) int
        mask_flag: 指定 mask 部分是 True 还是 False
        
        mask_lens 中的单个元素用位置(...,)确定后, 在原 tensor 中有唯一的长度为 d_size 的 1-D tensor 与其对应,
        mask_lens 元素值代表了这个 1-D tensor 从index 0(包括)开始, 有多少个是要 mask 的

        mask_lens 元素值可以为 0, 代表原 tensor 在该位置的长度为 d_size 的1-D tensor 所有都不需要mask.
    
    return:
        mask: (..., d_size)
        mask 是一个 True/False tensor, 它的 shape 等于 输入参数 tensor_shape.
    '''
    # index_1dtensor: [0, 1, ..., n_logits-1]
    index_1dtensor = torch.arange(tensor_shape[-1], device=mask_lens.device, dtype=mask_lens.dtype)

    # 伸展 mask_lens 的last dim, 即变成单个元素的 1d tensor. index_1dtensor 和 单个元素的1dtensor broadcast对比
    # [0, 1, ..., n_logits-1] with shape as (n_logtis, ) <?< [ ...,[valid_len_1], [valid_len_2], ..., ] with shape as (..., 1)
    mask = index_1dtensor < mask_lens.unsqueeze(-1)
    # braodcast 对比: [0, 1, ..., n_logits-1] 逐一 和 valid_len_i 对比, 得到 T/F mask shape as (..., n_logits)
    # 此时 mask 部分是 True, 非 mask 部分是 False. 符合 mask_flag 为 True 的情况.

    # 如果 mask_flag 为 False, 那么 mask 部分应该是False, 非 mask 部分为 True
    if not mask_flag:
        mask = ~mask

    return mask