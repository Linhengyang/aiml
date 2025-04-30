import torch.nn as nn
import torch



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
        the order of the sequence is first upper 'left to right', then move down 'left to right'
    '''
    # img_batch shape:(batch_size, num_channels, h, w)
    padImgBatch = minpad_to_divide(img_batch, patch_size, pad_mode, pad_value)
    # padImgBatch shape: (batch_size, num_channels, height+pad_num_h, width+pad_num_w)
    
    batch_size, num_chnls, _, _ = img_batch.shape
    p_h, p_w = patch_size

    # unfold padImgBatch: 当前 height+pad_num_h = height_paded 保证 整除 p_h, width+pad_num_w = width_paded 保证 整除 p_w
    # padImgBatch 
    #   shape: (B, _, height_paded, width_paded)
    #   stride: (B, _, width_paded, 1)

    # final result should be with 
    #   shape (B, height_paded//p_h * width_paded//p_w, _, , p_h, p_w)
    #   stride (B, p_h*p_w, _, , p_w, 1)


    # unfold(dim=3, p_w, p_w):
    # shape (B, _, height_paded, width_paded) -vanilla-> (B, _, height_paded, width_paded//p_w, p_w)
    # stride (B, _, width_paded, 1) -vanilla-> (B, _, width_paded, p_w, 1)

    # reshape to fusion dim2 and dim3
    # shape (B, _, height_paded, width_paded//p_w, p_w) --collapse dim2 and dim3--> (B, _, height_paded*width_paded//p_w, p_w)
    # stride (B, _, width_paded, p_w, 1) --collapse dim2 and dim3--> (B, _, p_w, 1)
    

    # continue to unfold(dim=2, p_h, p_h):
    # shape (B, _, height_paded*width_paded//p_w, p_w) -vanilla-> (B, _, height_paded//p_h*width_paded//p_w, p_h, p_w)
    #       -permute-> (B, _, height_paded//p_h*width_paded//p_w, p_w, p_h)

    # stride (B, _, p_w, 1) -vanilla-> (B, _, p_h*p_w, p_w, 1)
    #       -permute-> (B, _, p_h*p_w, 1, p_w)

    # permute(0, 2, 1, 4, 3):
    # shape(B, _, height_paded//p_h*width_paded//p_w, p_w, p_h) --> (B, height_paded//p_h*width_paded//p_w, _, p_h, p_w)
    # stride(B, _, p_h*p_w, 1, p_w) --> (B, p_h*p_w, _, p_w, 1)

    return padImgBatch.unfold(3, p_w, p_w).reshape(batch_size, num_chnls, -1, p_w).unfold(2, p_h, p_h).permute(0, 2, 1, 4, 3)