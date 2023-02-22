import torch.nn as nn
import torch
import numbers

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
    assert torch.is_tensor(img_batch) and img_batch.dim() == 4, 'input image batch should be tensor in shape (batch_size, num_channels, h, w)'
    h, w = img_batch.shape[2:]
    k_h, k_w = patch_size
    pad_num_h, pad_num_w = 0 if h % k_h == 0 else k_h - h % k_h, 0 if w % k_w == 0 else k_w - w % k_w
    pad_num_u = pad_num_h // 2
    pad_num_d = pad_num_h - pad_num_u
    pad_num_l = pad_num_w // 2
    pad_num_r = pad_num_w - pad_num_l
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

def patchify(img_batch, patch_size):
    '''
    input:
        1. img_batch: shape (batch_size, num_channels, h, w)
        2. patch_size: 2-D size of patch, (p_h, p_w)

    output:
        batch of sequence of patches, with shape:
        (batch_size, num_batches=seq_length, num_channels, patch_height, patch_width)
        the order of the sequence is first upper 'left to right', then move down 'left to right'
    '''
    # img_batch shape:(batch_size, num_channels, h, w)
    img_batch = minpad_to_divide(img_batch, patch_size)
    p_h, p_w = patch_size
    batch_size, num_chnls, _, _ = img_batch.shape
    return img_batch.unfold(2, p_h, p_h).unfold(3, p_w, p_w).reshape(batch_size, num_chnls, -1,p_h, p_w).transpose(1,2)