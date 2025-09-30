import torch.nn as nn
import torch
import torch.nn.functional as F


class MaskedCrossEntropyLoss(nn.CrossEntropyLoss):
    '''
    mask-version of CrossEntropyLoss
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor|None) -> torch.Tensor:
        '''
        input: (batch_size, num_cls, [d1...dk])float as logits
        target: (batch_size, [d1...dk])int64 as label
        mask: (batch_size, [d1...dk])bool as mask. only loss of true-area of target will be calculated
        '''
        # cross-entropy loss behavior:
        # pred(B, C, [d1...dk])float, label(B, [d1...dk])long --> unmasked_loss(B, [d1...dk])float
        # pred(B, C)float, label(B,)long --> unmasked_loss(B,)float
        # pred(C,)float, label(1,)long --> unmasked_loss(1,)float

        if mask is None:
            ignore_index = self.ignore_index
            labels = target
        else:
            ignore_index = -100
            labels = target.clone() # 防御性编程, 避免影响源数据
            labels[~mask] = ignore_index
        
        return F.cross_entropy(
            input,
            labels,
            weight = self.weight,
            ignore_index = ignore_index,
            reduction = self.reduction,
            label_smoothing = self.label_smoothing,
            )