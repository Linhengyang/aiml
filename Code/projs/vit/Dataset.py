from ...Utils.Text.TextPreprocess import preprocess_space
from ...Utils.Text.TextPreprocess import Vocab
from ...Utils.Common.SeqOperation import truncate_pad
import torch
import torch.nn as nn

def pad_to_divide(img_batch, patch_size):
    # img_batch shape:(batch_size, n_channels, h, w)
    h, w = img_batch.shape[2:]
    k_h, k_w = patch_size
    pad_num_h, pad_num_w = 0 if h % k_h == 0 else k_h - h % k_h, 0 if w % k_w == 0 else k_w - w % k_w
    pad_num_u = pad_num_h // 2
    pad_num_d = pad_num_h - pad_num_u
    pad_num_l = pad_num_w // 2
    pad_num_r = pad_num_w - pad_num_l
    return nn.functional.pad(img_batch, pad=(pad_num_l, pad_num_r, pad_num_u, pad_num_d))

def patchify(img_batch, patch_size):
    '''
    input:
        1. img_batch: shape (batch_size, num_channels, h, w)
        2. patch_size: 2-D size of patch, (p_h, p_w)

    output:
        batch of sequence of patches, shape:
        (batch_size, num_batches=seq_length, num_channels, patch_height, patch_width)
        the order of the sequence is first upper 'left to right', then move down 'left to right'
    '''
    # img_batch shape:(batch_size, num_channels, h, w)
    img_batch = pad_to_divide(img_batch, patch_size)
    p_h, p_w = patch_size
    batch_size, num_chnls, _, _ = img_batch.shape
    return img_batch.unfold(2, p_h, p_h).unfold(3, p_w, p_w).reshape(batch_size, num_chnls, -1,p_h, p_w).transpose(1,2)

class seq2seqDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        pass
    
    def __getitem__(self):
        pass
    
    def __len__(self):
        pass

