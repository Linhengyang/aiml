# Mask.py
import typing as t
import torch





def mask_first_n_valid(tensor_shape, valid_lens):
    '''
    input:
        tensor_shape: (..., n_logits)
        valid_lens: tensor with shape as (...,)
        
        valid_lens的某元素靠位置(...,)确定. 在原 tensor 中有唯一的长度为 n_logtis 的 1-D tensor 与其对应.
        valid_lens元素值代表了这个 1-D tensor 前几个是valid的. 元素值可以为 0, 这样在softmax运算之后, 得到平均分布
    
    return:
        mask: (..., n_logits)
        mask 是一个 True/False tensor, 它的shape 等于 输入参数 tensor_shape.
    '''
    # index_1dtensor: [0, 1, ..., n_logits-1]
    index_1dtensor = torch.arange(tensor_shape[-1], device=valid_lens.device, dtype=valid_lens.dtype)

    # 伸展 valid_lens 的last dim, 即变成单个元素的 1dtensor. index_1dtensor 在和 单个元素的1dtensor broadcast对比
    # [0, 1, ..., n_logits-1] with shape as (n_logtis, ) <?< [ ...,[valid_len_1], [valid_len_2], ..., ] with shape as (..., 1)
    mask = index_1dtensor < valid_lens.unsqueeze(-1)
    
    # braodcast 对比: [0, 1, ..., n_logits-1] 逐一 和 valid_len_i 对比, 得到 T/F mask shape as (..., n_logits)
    return mask