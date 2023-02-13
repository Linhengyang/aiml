import torch.nn as nn
import torch

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    '''
    About nn.CrossEntropyLoss
    input:
        1. pred, tensor of classification logits. Logit values on the second dim, batch_idx on the first dim
           shape: (batch_size, num_cls, [d1...dk] as position dims)
        2. label, tensor of classification labels. Label values are int32 values(not one-hot)
           shape: (batch_size, [d1...dk] as position dims)
    output:
        None-reduction CE Loss tensor w/ shape (batch_size, [d1...dk] as position dims). elements are loss
    
    About MaskedSoftmaxCELoss
    input:
        3. valid_lens, tensor of valid lens of label. Only losses from valid area will be considered.
           shape: (batch_size,)
    output:
        Loss tensor wight shape (batch_size,)
        Only losses from valid area are averaged for every sample.
        教科书这里有点问题, invalid tokens' losses 未被考虑, 但是在平均loss时却计入了这些invalid tokens,
        导致invalid tokens越多的序列sample, 其loss越小。
    '''
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, valid_len):
        '''
        pred: (batch_size, num_steps, num_cls)
        label: (batch_size, num_steps)
        '''
        self.reduction = 'none'
        # unmasked_loss shape: (batch_size, num_steps)
        # mask_mat shape: (batch_size, num_steps)
        unmasked_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0,2,1), label)
        def mask_matrix(X, valid_len):
            weights = torch.ones_like(X)
            maxlen = X.size(1)
            flags = torch.arange(maxlen, dtype=torch.float32, device=X.device).unsqueeze(0) < valid_len.unsqueeze(1)
            weights[~flags] = 0
            # weights /= weights.sum(dim=1, keepdim=True)
            # return weights.nan_to_num()
            return weights
        mask = mask_matrix(unmasked_loss, valid_len)
        # return (unmasked_loss * mask).sum(dim=1)
        return (unmasked_loss * mask).mean(dim=1)