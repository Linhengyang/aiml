import torch.nn as nn
import torch
from torch import Tensor
from typing import Optional

class BPRankingLoss(nn.Module):
    '''
    args:
        1. reduction: one of ('sum', 'mean', 'none'), default as 'mean's
        2. weight: Tensor(Optional), if given has to be a Tensor to give weight for every loss
    input:
        input1 & input2 & target: 1-D minibatch or 0-D tensor
        target: (containing 1 or -1), '1' means input1 rank higher than input2, '-1' means input2 rank higher than input 1
    output:
        loss(x1, x2, y) = - log( y * sigmoid(x1-x2) ) if reduction = 'none' & weight = None
        return mean of loss(scalar) if reduction = 'mean', sum of loss(scalar) is reduction = 'sum'
    '''
    reduction: str
    __constants__ = ['reduction']

    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        super(BPRankingLoss, self).__init__()
        self.reduction = reduction
        self.register_buffer('weight', weight)
        self.weight: Optional[Tensor]

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        if (input1.dim() != input2.dim() or input1.dim() != target.dim()):
            raise RuntimeError(
                (
                    "bayesian_personalized_ranking_loss : All input tensors should have same dimension but got sizes: "
                    "input1: {}, input2: {}, target: {} ".format(input1.size(), input2.size(), target.size())
                )
            )
        _loss = -torch.log(torch.sigmoid(target*(input1-input2)))
        if self.weight:
            _loss = torch.mul(self.weight, _loss)
        if self.reduction == 'none':
            return _loss
        elif self.reduction == 'mean':
            return _loss.mean()
        elif self.reduction == 'sum':
            return _loss.sum()
        else:
            raise ValueError("{} is not a valid value for reduction".format(self.reduction))