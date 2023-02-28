import torch.nn as nn
import torch
import math
from ...Modules._recsys import MaskedMatrixFactorization, UnMaskMatrixFactorization

class MatrixFactorization(nn.Module):
    def __init__(self, num_factors, num_users, num_items):
        super().__init__()
        self.mf = UnMaskMatrixFactorization(num_factors, num_users, num_items)
    
    def forward(self, users_idx, items_idx):
        # shapes: (batch_size,)int64, (batch_size,)int64
        return self.mf(users_idx, items_idx)
    
    def __str__(self):
        return self.mf.__str__()

class ItemBasedAutoRec(nn.Module):
    def __init__(self, num_hiddens, num_users, dropout=0.05):
        super().__init__()
        self.encoder = nn.Linear(num_users, num_hiddens)
        self.act = nn.Sigmoid()
        self.decoder = nn.Linear(num_hiddens, num_users)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_batch_matrix):
        # input shape: (batch_size=num_items_batch, num_users)
        pred = self.decoder(self.dropout(self.act(self.encoder(input_batch_matrix))))
        if self.training:
            return pred * torch.sign(input_batch_matrix)
        else:
            return pred