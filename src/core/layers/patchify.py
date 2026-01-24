import numbers
import torch
from torch import nn



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





class Patchify(nn.Module):
    '''
    args:
        img_shape: (num_channels, h, w)
        patch_size: Integral or tuple of two integrals
        pad_mode: default 'constant'
        pad_value: default 0
    
    inputs: img_batch
        img_batch: (batch_size, num_channels, h, w)

    returns: batch of sequence of patches, with shape:
        (batch_size, num_patches=seq_length, num_channels, patch_height, patch_width)

        the order of the sequence is first move down, then move right, 
        which is 1 -> 4 -> 7 -> 2 -> 5 -> 8 -> 3 -> 6 -> 9
        for  1  2  3
             4  5  6
             7  8  9
    
    attributes:
        patch_size
        num_patches
        patch_flatlen
    '''
    def __init__(self, img_shape, patch_size, pad_mode='constant', pad_value=0):

        super().__init__()
        if isinstance(patch_size, numbers.Integral):
            patch_size = (patch_size, patch_size)
        
        self.img_shape = img_shape
        self._patch_size, self.pad_mode, self.pad_value = patch_size, pad_mode, pad_value
        self._num_patches, self._patch_flatlen = calc_patchifed_sizes(img_shape, patch_size)


    def forward(self, img_batch):
        # 确保 img_batch 的 shape 和 参数 img_shape 相同. 
        # 这里 img_batch 作为 tensor, .shape 返回 torch.Size dtype. 但是 self.img_shape 可能是 list/tuple 等
        assert img_batch.shape[1:] == torch.Size(self.img_shape), \
            f"image batch shape {img_batch.shape[1:]} not match with argument img_shape {self.img_shape}"
        
        return patchify(img_batch, self._patch_size, self.pad_mode, self.pad_value)
    
    @property
    def patch_size(self):
        return self._patch_size

    @property
    def num_patches(self):
        return self._num_patches
    
    @property
    def patch_flatlen(self):
        return self._patch_flatlen
    