import math
import torch
from torch import nn

class AddLNorm(nn.Module):
    '''
    args: norm_shape, dropout
        norm_shape: the length of tensor's dim for layer norm
        dropout: dropout rate. Regularization on second input
    
    inputs: X, f_X
        X: minibatch
        f_X: MLP(X) with same shape as X. Usually to achieve resildual connection

    returns: denoted as O
        O: same shape with input X
    
    explains:
        Add(usually for residual connection), then layer normalize
    '''
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)
    
    def forward(self, X, f_X):
        assert X.shape[1:] == f_X.shape[1:], 'Adding two inputs with different shape'
        return self.ln(X + self.dropout(f_X))