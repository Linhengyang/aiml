import math
import torch
import numbers
from torch import nn

class Patchify(nn.Module):
    '''
    args: img_shape, patch_size, pad_mode, pad_value
        img_shape: (num_channels, h, w)
        patch_size: Integral or tuple of two integrals
        pad_mode: 'constant'
        pad_value: 0
    
    inputs: img_batch
        img_batch: (batch_size, num_channels, h, w)

    returns: batch of sequence of patches, with shape:
        (batch_size, num_batches=seq_length, num_channels, patch_height, patch_width)
        the order of the sequence is first upper 'left to right', then move down 'left to right'
    
    attributes:
        patch_size
        num_patches
        patch_flatlen
    '''
    def __init__(self, img_shape, patch_size, pad_mode='constant', pad_value=0):
        super().__init__()
        if isinstance(patch_size, numbers.Integral):
            patch_size = (patch_size, patch_size)
        self._patch_size = patch_size
        k_h, k_w = patch_size
        assert len(img_shape) == 3, 'image shape should be (num_channels, h, w)'
        c, h, w = img_shape
        pad_num_h, pad_num_w = 0 if h % k_h == 0 else k_h - h % k_h, 0 if w % k_w == 0 else k_w - w % k_w
        self._num_patches = int( (h+pad_num_h)*(w+pad_num_w)/(k_h * k_w) )
        self._patch_flatlen = int( c * k_h * k_w )
        self.pad_num_u, self.pad_num_l = pad_num_h // 2, pad_num_w // 2
        self.pad_num_d, self.pad_num_r = pad_num_h - self.pad_num_u, pad_num_w - self.pad_num_l
        self.pad_mode, self.pad_value = pad_mode, pad_value

    def forward(self, img_batch):
        padImgBatch = nn.functional.pad(img_batch, pad=(self.pad_num_l, self.pad_num_r, self.pad_num_u, self.pad_num_d),
                                        mode=self.pad_mode, value=self.pad_value)
        batch_size, num_chnls, _, _ = img_batch.shape
        p_h, p_w = self._patch_size
        return padImgBatch.unfold(2, p_h, p_h).unfold(3, p_w, p_w).reshape(batch_size, num_chnls, -1,p_h, p_w).transpose(1,2)
    
    @property
    def patch_size(self):
        return self._patch_size

    @property
    def num_patches(self):
        return self._num_patches
    
    @property
    def patch_flatlen(self):
        return self._patch_flatlen