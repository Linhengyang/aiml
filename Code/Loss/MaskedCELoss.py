import torch.nn as nn
import torch

class MaskedSoftmaxCELoss_deprecate(nn.CrossEntropyLoss):
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
    
    About MaskedSoftmaxCELoss
    input:
        1. & 2. are inheritting from nn.CrossEntropyLoss
        3. valid_lens, tensor of valid lens of pred/label. Only losses from valid area will be considered.
           shape: (batch_size,)
    output:

        sum on dim 1. Loss tensor wight shape (batch_size,)
        Only losses from valid area are summed for every sample.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred, label, valid_lens):
        '''
        pred: (batch_size, num_steps, num_cls)logits
        label: (batch_size, num_steps)int64
        valid_lens: (batch_size, )int32
        '''

        self.reduction = 'none'
        # unmasked_loss shape: (batch_size, num_steps) with float elements
        unmasked_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0,2,1), label)

        # mask_mat shape: (batch_size, num_steps) with 0-1 elements where 1 indicates valid
        def mask_matrix(X, valid_lens):
            weights = torch.ones_like(X)
            maxlen = X.size(1)
            # [[0, 1, 2, ... num_steps-1]] < [ [vlens_0], [vlens_1], ..., [vlens_N] ]
            # (-1, num_steps) broadcast with (batch_size, 1) --> (batch_size, num_steps)
            flags = torch.arange(maxlen, dtype=torch.float32, device=X.device).unsqueeze(0) < valid_lens.unsqueeze(1)
            weights[~flags] = 0 # invalid part set to 0
            return weights
        
        mask = mask_matrix(unmasked_loss, valid_lens)

        # sum on dim 1 of (batch_size, num_steps) --> (batch_size,)
        return (unmasked_loss * mask).sum(dim=1)
    






















class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
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
    
    About MaskedSoftmaxCELoss
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
        unmasked_loss = super(MaskedSoftmaxCELoss, self).forward(pred, label)

        # valid_area shape: (batch_size, [d1...dk] as position dims) with 0-1/True-False elements where 1/True indicates valid
        # sum on dims other than 0 of (batch_size, [d1...dk] as position dims) --> (batch_size,)

        return (unmasked_loss * valid_area.to(dtype=torch.int32)).sum( dim=tuple(range(1, valid_area.ndim)) )