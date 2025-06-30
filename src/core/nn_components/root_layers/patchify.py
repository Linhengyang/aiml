import numbers
import torch
from torch import nn
from ...base.functions.patch_operation import patchify, calc_patchifed_sizes

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
    