import torch.nn as nn
import torch


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

    def forward(self, pred, label, valid_area):
        '''
        pred: (batch_size, num_cls, [d1...dk] as position dims)float logits
        label: (batch_size, [d1...dk] as position dims)int64
        valid_area: (batch_size, [d1...dk] as position dims)0-1(int32)/True-False(bool)
        '''

        self.reduction = 'none'

        # unmasked_loss shape: (batch_size, [d1...dk] as position dims) with float logits
        unmasked_loss = super(MaskedCrossEntropyLoss, self).forward(pred, label)

        # valid_area shape: (batch_size, [d1...dk] as position dims) with 0-1/True-False elements where 1/True indicates valid
        # sum on dims other than 0 of (batch_size, [d1...dk] as position dims) --> (batch_size,)

        return (unmasked_loss * valid_area.to(dtype=torch.bool)).sum( dim=tuple(range(1, valid_area.ndim)) )