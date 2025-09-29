import torch.nn as nn
import torch
from PIL import Image


def minpad_to_divide(img_batch, patch_size, mode='constant', value=0):
    '''
    input:
        1. img_batch: shape (batch_size, num_channels, h, w)
        2. patch_size: 2-D size of patch, (p_h, p_w)

    output:
        batch of images with unchanged num_channels & batch_size
        but height & width are pad in minimum to perfect patch_size sized patch dividing
    
    explain:
        输出的img_batch在尺寸上被pad调整, 可以被patch_size完美patch化
    '''
    # img_batch shape:(batch_size, n_channels, h, w)
    assert torch.is_tensor(img_batch) and img_batch.dim() == 4,\
        f'input image batch should be tensor in shape (batch_size, num_channels, h, w)'
    
    h, w = img_batch.shape[2:]
    k_h, k_w = patch_size

    # 若 img height 不能整除 kernel height, 那么 pad img 的 height 到 h+pad_num_h, 这里 pad_num_h = kernel_h - h % kernel_h, 保证可整除
    # 若 img width 不能整除 kernel width, 那么 pad img 的 width 到 w+pad_num_w, 这里 pad_num_w = kernel_w - w  % kernel_w, 保证可整除
    pad_num_h, pad_num_w = 0 if h % k_h == 0 else k_h - h % k_h, 0 if w % k_w == 0 else k_w - w % k_w

    # pad 的方式是 四周 均匀 pad: 
    # 上下总共pad pad_num_h, 上面 pad 数量 pad_num_h // 2
    # 左右总共pad pad_num_w, 左边 pad 数量 pad_num_w // 2
    pad_num_u, pad_num_l = pad_num_h // 2, pad_num_w // 2
    pad_num_d, pad_num_r = pad_num_h - pad_num_u, pad_num_w - pad_num_l

    return nn.functional.pad(img_batch, pad=(pad_num_l, pad_num_r, pad_num_u, pad_num_d), mode=mode, value=value)



def calc_patchifed_sizes(img_shape, patch_size):
    '''
    img_shape: (c, h, w), patch_size: (p_h, p_w)
    计算完美patch化后, patch个数和每个patch_flatten长度
    '''
    c, h, w = img_shape
    k_h, k_w = patch_size
    pad_num_h, pad_num_w = 0 if h % k_h == 0 else k_h - h % k_h, 0 if w % k_w == 0 else k_w - w % k_w
    num_patches = int( (h+pad_num_h)*(w+pad_num_w)/(k_h * k_w) )
    patch_flatlen = c * k_h * k_w

    return num_patches, patch_flatlen



def patchify(img_batch, patch_size, pad_mode="constant", pad_value=0):
    '''
    input:
        1. img_batch: tensor with shape (batch_size, num_channels, h, w)
        2. patch_size: 2-D size of patch, (patch_height, patch_width)

    output:
        batch of sequence of patches, with shape:
        (batch_size, num_batches=seq_length, num_channels, patch_height, patch_width)

        the order of the sequence is first move down, then move right, 
        which is 1 -> 4 -> 7 -> 2 -> 5 -> 8 -> 3 -> 6 -> 9
        for  1  2  3
             4  5  6
             7  8  9
    '''
    # img_batch shape:(batch_size, num_channels, h, w)
    padImgBatch = minpad_to_divide(img_batch, patch_size, pad_mode, pad_value)
    # padImgBatch shape: (batch_size, num_channels, height+pad_num_h, width+pad_num_w)
    
    batch_size, num_chnls, H, W = padImgBatch.shape
    p_h, p_w = patch_size

    # unfold padImgBatch: 当前 height+pad_num_h = height_paded = H 保证 整除 p_h, width+pad_num_w = width_paded = W 保证 整除 p_w
    # padImgBatch 
    #   shape: (B, 3, H, W)
    #   stride: (_ = 3*H*W, __ = H*W, W, 1)


    # unfold(dim=3, p_w, p_w):
    # shape (B, 3, H, W) -vanilla-> (B, 3, H, W//p_w, p_w)
    # stride (3*H*W, H*W, W, 1) -vanilla-> (3*H*W, H*W, W, p_w, 1)
    padImgBatch_unfold_W = padImgBatch.unfold(3, p_w, p_w)


    # switch unfolded dim(width) forward
    padImgBatch_unfold_W = padImgBatch_unfold_W.permute(0, 1, 3, 2, 4)
    # shape(B, 3, W//p_w, H, p_w), stride(3*H*W, H*W, p_w, W, 1)

    # continue to unfold(dim=3, p_h, p_h):
    # shape (B, 3, W//p_w, H, p_w) -vanilla-> (B, 3, W//p_w, H//p_h, p_h, p_w)
    #       -permute-> (B, 3, W//p_w, H//p_h, p_w, p_h)
    # stride (3*H*W, H*W, p_w, W, 1) -vanilla-> (3*H*W, H*W, p_w, W*p_h, W, 1)
    #       -permute-> (3*H*W, H*W, p_w, W*p_h, 1, W)
    padImgBatch_unfold_W_H = padImgBatch_unfold_W.unfold(3, p_h, p_h)

    # permute(0, 1, 2, 3, 5, 4):
    # shape (B, 3, W//p_w, H//p_h, p_w, p_h) --> (B, 3, W//p_w, H//p_h, p_h, p_w)
    # stride (3*H*W, H*W, p_w, W*p_h, 1, W) --> (_, __, p_w, W*p_h, W, 1)
    padImgBatch_unfold_W_H = padImgBatch_unfold_W_H.permute(0,1,2,3,5,4)

    # permute(0, 2, 3, 1, 4, 5):
    # shape (B, 3, W//p_w, H//p_h, p_h, p_w) --> (B, W//p_w, H//p_h, 3, p_h, p_w)
    # stride (3*H*W, H*W, p_w, W*p_h, W, 1) --> (3*H*W, p_w, W*p_h, H*W, W, 1)
    padImgBatch_unfold_W_H = padImgBatch_unfold_W_H.permute(0,2,3,1,4,5)

    # reshape(B, W//p_w * H//p_h, 3, p_h, p_w):
    # shape (B, W//p_w, H//p_h, 3, p_h, p_w) --> (batch_size, W//p_w * H//p_h, 3, p_h, p_w)
    # stride (3*H*W, p_w, W*p_h, H*W, W, 1) --> (3*H*W, W*p_h, H*W, W, 1)

    # 省略了 width 分割之间的 stride(因为 width 之间的分割，跨度为 p_w)
    # 不能用 view, 因为 view 不支持 stride 交错. 不能用 as_strided(), 它也不满足交错. reshape 可以正确摊平 W 和 H 分割创造的维度
    padImgBatch_unfold_output = padImgBatch_unfold_W_H.reshape(batch_size, W//p_w * H//p_h, num_chnls, p_h, p_w)

    # output
    # shape: (batch_size, W//p_w * H//p_h, 3, p_h, p_w)
    # stride: (3*H*W, W*p_h, H*W, W, 1)
    # (batch之间要跨越所有像素, patch之间要跨越总行像素*patch行数, chnl之间要跨越总2D像素, patch内部跨行要跨越总行像素, patch行内跨域是连续的)

    return padImgBatch_unfold_output





# tensor.unfold(dimension, size, step) 方法
# 一个 tensor的存储:
# 1 meta-data, 一维连续的内存存储
# 2 shape, 定义 tensor 的维度大小, 比如 (4, 3, 2) 代表在维度 (0, 1, 2) 的大小: 
#   维度0上有4个元素, 
#   给定维度0, 每个元素在维度1上有3个子元素
#   给定维度0和维度1, 每个子元素在维度2上有2个子子元素。这里子子元素就是scalar.

# 这里有一个“给定维度0”的说法, 原因是如果不考虑维度0, 维度1总长度有4*3=12。但实际上, tensor在维度1上不存在多于3的长度, 因为多余3的长度, 是以 n*3+m 决定的
# 同样的, 不考虑维度0和1, 维度2总长度有24. 但实际上, tensor在维度2上不存在多余2的长度, 因为多余2的长度, 是以 p*6 + q*3 + k 决定的.

# 倒序: 
#   维度2有两个scalar元素的位置, 组成一个长度为2的组合元素.
#   维度1上有3个组合元素的位置, 组成一个长度为3的组组合元素.
#   维度0上有4个位置, 4个组组合组成当前tensor



# 3 strides, 定义 tensor 的索引寻址偏移, 比如(6, 2, 1) 代表在维度 (0, 1, 2)的寻址偏移量: 
#   给定维度1和2, 在维度0上移动一个单位, 需要在meta-data中偏移6。维度0上最多可移动4-1=3个单位
#   给定维度0和2, 在维度1上移动一个单位, 需要在meta-data中偏移2。维度1上最多可移动3-1=2个单位
#   给定维度0和1, 在维度2上移动一个单位, 需要在meta-data中偏移1。维度2上最多可移动2-1=1个单位

#   一般来说, strides[-1] 要保持 等于 1, 保证 tensor 在最后一个维度上的偏移 等价于 在 meta-data 中的偏移


## 
# 利用 strides 和 在meta-data上的总偏移 num_steps 确定 index 位置:
#   num_steps % strides[0] 得到的是 dimension 0 的 index, 余数 num_steps // strides[0] = num_step_expt_dim0
#   num_step_expt_dim0 % strides[1] 得到的是 dimension 1 的 index, 余数 num_step_expt_dim0 // strides[1] = num_step_expt_dim01
#   。。。
#   num_step_expt_ % strides[-1] 得到的是 dimension -1 (最后一个维度) 的 index


## 对于 non-overlapping 的 tensor data, shape 和 strides 之间存在一定的数字关系
# 1. strides[-1] 永远等于 1
# 2. strides[-2] 等于 shape[-1], 因为 strides[-2] 代表了 在倒数第二维度偏移一个单位, 需要 meta-data 上的偏移量。
#    在 non-overlapping 的前提下, 这个量就代表了 在倒数第一维度最多允许的偏移量，即 倒数第一维度的长度.
# 3. strides[-3] 等于 shape[-2] * shape[-1], 因为 strides[-3] 代表了 在倒数第三维度偏移一个单位, 需要 meta-data 上的偏移量。
#    在 non-overlapping 的前提下, 这个量代表偏移 最多倒数第二维度长度次偏移, 而每一次倒数第二维度上的偏移，都代表了 倒数第一维度长度次偏移。
#    故 得到了 shape[-2] * shape[-1] 的结果。
# 4. 其他剩余维度的关系类似: 在 non-overlapping 的前提下, strides 和 shape 之间存在累乘的关系


# 但是在通用的视角下, 基于 meta-data 之上的 shape 和 stride 不存在数字关系: shape 是 shape, stride 是 stride,
# shape 代表 各个 dimension, 在给定 前面低维 dimension 之后, 在当前维度 可以最多有多少次偏移量, 即为当前维度的 length-1
# 比如 tensor X 的shape 是 (d0, d1, d2, d3, d4), 代表在 dim0 可以最多有 d0-1 次偏移(d0个位置); 选定 dim0 后, 在 dim1 可以最多有 d1-1 次偏移(d1个位置)

# stride 代表 各个dimension, 在当前维度 偏移一个单位（其他维度不偏移），需要在 meta-data 上等效移动 多少个单位, 即为当前维度的 stride
# 比如 tensor X 的stride 是 (s0, s1, s2, s3, s4), 代表在 dim0 一次偏移(其他dim不动), 等价于在 meta-data 上偏移 s0 个单位
# 可以看出 stride 只需要满足 都是 正整数 即可. 各维度上的  stride 数字之间 没有任何 限制关系: 即
# 给定 meta-data, 给定 shape 和 stride(相同长度), shape 和 stride 除了正整数外无其他要求, 即可从 meta-data 中寻址, 构建满足的 tensor





# unfold(dimension, size, step) 方法 不改变 tensor 的 meta-data, 通过在维度 dimension 上作 宽度为 size 的滑动窗口, 并以 step 为跳跃步长滑动
# 对于一个 shape 为 (d_0, d_1, ..., d_i, ..., d_n), stride 为 (s_0, s_1, ..., s_i, ..., s_n). 现在要 unfold dimension i
# unfold(dimension = d_i, size = window_size, step = step), 那么:
# 对于 d_i 前面的所有维度来说, 即 d_0, ..., d_i-1, 没有变化。即本来是 d_0, ..., d_i-1 各维度 index 都确定后, 得到一个 (d_i,...,d_n) 的tensor, 
# 只是现在这个 (d_i, ..., d_n) 的 tensor 要被 unfold 成另一个形状了。
# 所以 前面维度的shape d_0, ..., d_i-1 和 前面维度的stride s_0, ..., s_i-1 都没有变化。

# unfold 的关键在于, 原来shape 为 (d_i,...,d_n), stride 为 (s_i,...,s_n) 的 tensor, 要如何作 窗口宽度为 window_size, 滑动跨度为 step 的 unfold 操作?





# 1、 以 d_i 为总长度（滑动过程中窗口不越界） 作 窗口宽度为 window_size、滑动步长为 step 的滑动，那么总共会产生 
# [ (d_i - window_size)/step ] + 1 = num_windows 个窗口
#       (显然 d_i 维度中最后几个维度可能不会被滑动窗口覆盖, 故可能在新tensor中消失)

# 原 tensor 在第 i 维上, stride 是 s_i, 长度是 d_i 个 shape为(d_i+1,...d_n)/stride为(s_i+1,...s_n) 的 原sub-tensor

# 新 tensor 在第 i 维上, stride 是 s_i * step
#       (原来 d_i 维, 每偏移一个单位, 在meta-data上偏移s_i; 现在是 num_windows 维, window 偏移到下一个 window, 跳过了 step 个原单位, 故新的stride = s_i*step)

# 长度是 num_windows 个 
#   "由 window_size 个 原sub-tensor, 即 shape为(window_size, d_i+1,...d_n)/stride为(s_i, s_i+1,...s_n) 的tensor permute 的 新sub-tensor"


# 有点绕, 首先总结 新 tensor 在第 i 维上, stride 是 s_i * step, 长度是 num_windows 个 新 sub-tensor.

# 其次, 考虑不 permute 的unfold方法, 那么当 unfold 在 第i维, 过程应该是:
# shape(d_i,...,d_n) / stride(s_i,...,s_n)  ------------unfold on dimension i------------> 
# shape(num_windows, window_size, d_i+1,...d_n) / stride(s_i*step, s_i, s_i+1,...s_n)
# 这个过程很好理解, 即 d_i 维被拆分成了两个维度, 长度分别是 num_windows, window_size，stride 分别是 s_i*step, s_i

# 但是 torch.unfold 对 unfold 的结果做了一个 额外的 permute:
# shape(num_windows, window_size, d_i+1,...d_n) / stride(s_i*step, s_i, s_i+1,...s_n)  ------------permute------------>
# shape(num_windows, d_i+1,...d_n, window_size) / stride(s_i*step, s_i+1,...s_n, s_i)
# 即把 新增的 窗口内维度 permute 到最后一维度

# torch.unfold 总结:
# unfold on dimension i with window size as size, step as step:  
# shape(d0,..., d_i-1, d_i,  d_i+1,...,d_n) / stride(s_0,..., s_i-1, s_i,  s_i+1,...,s_n)   ------vanilla unfold on dim i------>
# shape(d0,..., d_i-1, num_windows, window_size, d_i+1,...d_n) / stride(s_0,..., s_i-1, s_i*step, s_i, s_i+1,...s_n)  ------permute last dim------>
# shape(d0,..., d_i-1, num_windows, d_i+1,...d_n, window_size) / stride(s_0,..., s_i-1, s_i*step, s_i+1,...s_n, s_i)