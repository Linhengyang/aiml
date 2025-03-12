# Mask.py
import typing as t
import torch



def valid_slice_mask(shape, valid_lens):
    '''
    input:
        shape: (batch_size, n_queries, n_logits)

        valid_lens: (batch_size, n_queries) or (batch_size,)
        valid_lens代表了各条 query 有多少 logits 参与 softmax 运算. valid_lens 可以为 0, 这样在softmax运算之后, 得到平均分布
    
    return:
        mask: (batch_size, n_queries, n_logits)
        mask 是一个 True/False tensor, 其 batch_size = i, n_queries = j 处的 query, 总共 n_logits 个logit中, 只有 前 p 个参与概率分布生成(softmax)
        p = valid_lens[i, j] 或 valid_lens[i] 取决于 valid_lens 是 2D 还是 1D tensor. 
    '''
    # shape = (batch_size, n_queries, n_logits)

    # 如果 valid_lens 是一个 1D tensor: (batch_size, ) -> (batch_size, n_queries)
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave( valid_lens.unsqueeze(1), repeats=shape[1], dim=1)
    elif valid_lens.dim() != 2:
        raise ValueError(
            f'input param valid_lens with wrong shape {valid_lens.shape}. must be 2D or 1D tensor')
    else:
        pass

    # [0, 1, ..., n_logits-1] 的 index tensor. 直接创建在 valid_lens 所在的设备上, 且使用 valid_lens 所使用的数据类型
    index_ts = torch.arange(shape[-1], device=valid_lens.device, dtype=valid_lens.dtype)

    # index_ts: (n_logits, ), valid_lens: (batch_size, n_queries)
    mask = index_ts < valid_lens.unsqueeze(2)

    return mask




def frontValidMask(shape, valid_lens):
    '''
    input:
        shape: (..., n_logits) 
        valid_lens: tensor with shape as (...,)
        
        valid_lens的某元素有位置(...,), 在原shape中有唯一的长度为n_logtis的1dtensor与其对应. 元素值代表了这个1dtensor前几个是valid的
    
    return:
        mask: (..., n_logits)
        mask 是一个 True/False tensor, 它的shape等于 输入参数shape. 前提是 valid_lens 只比
    '''

    index_1dtensor = torch.arange(shape[-1], device=valid_lens.device, dtype=valid_lens.dtype)

    # 伸展 valid_lens 的last dim, 即变成单个元素的1dtensor. index_1dtensor 在和 单个元素的1dtensor对比过程中, broadcast机制保证生成结果是 shape
    mask = index_1dtensor < valid_lens.unsqueeze(-1)

    return mask


















































if __name__ == "__main__":

    ## (batch_size, n_queries, n_logits)
    shape = (3, 4, 5)

    ## (batch_size, n_queries)
    valid_lens = torch.tensor([[3, 4, 1, 0],
                               [2, 1, 0, 1],
                               [1, 4, 5, 5]])
    
    ## (batch_size,)
    valid_lens = torch.tensor([1, 3, 5])

    print(valid_lens.dtype)

    print(valid_slice_mask(shape, valid_lens))