import torch.nn as nn
import torch

class pairBPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        raise NotImplementedError

class pairRankHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        raise NotImplementedError