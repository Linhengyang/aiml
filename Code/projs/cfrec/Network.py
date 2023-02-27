import torch.nn as nn
import torch
import math
from ...Modules._cfrec import MaskedMatrixFactorization, UnMaskMatrixFactorization

class explicitCF(nn.Module):
    def __init__(self, num_factors, num_users, num_items):
        super().__init__()
        self.mf = UnMaskMatrixFactorization(num_factors, num_users, num_items)
    
    def forward(self, users_idx, items_idx):
        # shapes: (batch_size,)int64, (batch_size,)int64
        return self.mf(users_idx, items_idx)
    
    def __str__(self):
        return self.mf.__str__()