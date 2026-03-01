# z-loss from PaLM
# 当 logits[batch_size, num_cls, [d1...dk]] 与 labels[batch_size, [d1...dk]] 贡献 cross-entropy Loss 时,
# 为了抑制 logits 的正向漂移, 促使 logits 分布在 0 以下附近的 float-comfortable 区域, 额外计算一个
# z-loss = square of log-sum-exp of logits 

from torch.nn.modules.loss import _Loss
from torch import Tensor
import torch

class ZLoss(_Loss):

    __constants__ = ["reduction"]

    def __init__(self, alpha: float, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)
        self.alpha = alpha

    def forward(self, logits: Tensor, mask: Tensor|None) -> Tensor:
        """
        logits: (batch_size, num_cls, [d1...dk])float
        mask: None or (batch_size, [d1...dk])bool as mask. only true-area of target attend loss
        """
        zloss = torch.logsumexp(logits, dim=1, keepdim=False) # [batch_size, [d1...dk]]
        if mask is None:
            if self.reduction == 'sum':
                return self.alpha * torch.sum(zloss)
            elif self.reduction == 'mean':
                return self.alpha * torch.mean(zloss)
            else:
                return self.alpha * zloss
        else:
            # mask is not None. False area not attend to loss 
            zloss[~mask] = 0
            if self.reduction == 'sum':
                return self.alpha * torch.sum(zloss)
            elif self.reduction == 'mean':
                return self.alpha * torch.sum(zloss) / torch.sum(mask)
            else:
                return self.alpha * zloss