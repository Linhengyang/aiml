import torch.nn as nn
import torch
import torch.nn.functional as F


class MaskedCrossEntropyLoss(nn.CrossEntropyLoss):
    '''
    About nn.CrossEntropyLoss
    input:
        1. pred, tensor of classification logits. batch_idx on the first dim, logit values on the second dim, following dims are position dims
           shape: (batch_size, num_cls, [d1...dk] as position dims)
        2. label, tensor of classification labels. Label values are int64 values(not one-hot)
           shape: (batch_size, [d1...dk] as position dims)
    output:
        if reduction set to 'none':
        None-reduction CE Loss tensor w/ shape (batch_size, [d1...dk] as position dims). element is loss value for single classification
    
    About MaskedCrossEntropyLoss
    input:
        1. & 2. are inheritting from nn.CrossEntropyLoss
        3. mask for valid_area, 0-1/True-False tensor of valid area of pred & label. Only losses from valid area will be calculated.
           shape: (batch_size, [d1...dk] as position dims), same as label
    output:

        sum on dims other than first dim. Output Loss tensor shape (batch_size,)
        Only losses from valid area are summed for every sample.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor|None) -> torch.Tensor:
        '''
        input: (batch_size, num_cls, [d1...dk])float as logits
        target: (batch_size, [d1...dk])int64 as label
        mask: (batch_size, [d1...dk])bool as mask. only count loss of true-area.
        '''

        # pred(B, C, [d1...dk])float, label(B, [d1...dk])long --> unmasked_loss(B, [d1...dk])float
        # pred(B, C)float, label(B,)long --> unmasked_loss(B,)float
        # pred(C,)float, label(1,)long --> unmasked_loss(1,)float
        unmasked_loss = F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        if mask is None:
            _loss = unmasked_loss
        else:
            _loss = unmasked_loss.masked_fill((mask==False), 0.)
        
        if self.reduction == 'none':
            return _loss
        elif self.reduction == 'mean':
            return _loss.mean()
        elif self.reduction == 'sum':
            return _loss.sum()
        else:
            raise ValueError(f'wrong arg `reduction` {self.reduction}. must be one of `mean`/`sum`/`none`.')