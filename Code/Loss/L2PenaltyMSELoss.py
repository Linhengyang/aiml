import torch
import torch.nn as nn

class L2PenaltyMSELoss(nn.MSELoss):
    '''
    About nn.MSELoss
    input:
        1. pred_tensor, same shape with target_tensor
        2. truth_tensor, same shape with pred_tensor
    output:
        None-reduction MSE Loss tensor w/ the same shape of input tensors with reduction = 'none'
    
    About L2PenaltyMSELoss
    input:
        3. lambd: penalty coefficient
        4. weight_tensors: length-changable args, consisting param tensors
    output:
        a scalar, sum of mse loss(reduction='sum') and l2 penalty
    '''
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred_tensor, truth_tensor, *weight_tensors):
        self.reduction = 'sum'
        mse_loss = super(L2PenaltyMSELoss, self).forward(pred_tensor, truth_tensor)
        l2_penalty = 0
        for weight_tensor in weight_tensors:
            l2_penalty += torch.sum(weight_tensor.pow(2)) / 2
        l2_penalty = self.lambd * l2_penalty
        return mse_loss + l2_penalty