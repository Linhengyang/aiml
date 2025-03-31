import torch.nn as nn
import torch

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    '''
    About nn.CrossEntropyLoss
    input:
        1. pred, tensor of classification logits. Logit values on the second dim, batch_idx on the first dim
           shape: (batch_size, num_cls, [d1...dk] as position dims)
        2. label, tensor of classification labels. Label values are int64 values(not one-hot)
           shape: (batch_size, [d1...dk] as position dims)
    output:
        if reduction set to 'none':
        None-reduction CE Loss tensor w/ shape (batch_size, [d1...dk] as position dims). element is loss value for single classification
    
    About MaskedSoftmaxCELoss
    input:
        1. 2. inheritting from nn.CrossEntropyLoss
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