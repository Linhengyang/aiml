import math
import torch
from torch import nn

class TrigonoPosEnc(nn.Module):
    '''
    args: num_hiddens, dropout
        num_hiddens: the feature dimentions of input X, simply as d
        dropout: dropout rate. Regularization on final X + PosEnc(X)
    
    inputs: X
        X's shape: (batch_size, seq_len, d=num_hiddens)
    
    returns: denoted as S (in-place, same device with input X)
        S's shape: (batch_size, seq_len, d=num_hiddens)

    explains:
        When input X (batch_size, seq_len, d) arrives, Sinusoidal positional encoding P with shape (seq_len, d)
        is added to every sample of X. P with shape (seq_len, d), whose
            (k, 2j) element is sin( k/10000^(2j/d) )
            (k, 2j+1) element is cos( k/10000^(2j/d) )
    '''
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens)) # create a long enough P with same d as input X
        # X 2-D tensor, 行idx是0到max_len-1, 列idx是0到num_hiddens-1之内的偶数
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000,
            torch.arange(0, num_hiddens, 2, dtype=torch.float32).reshape(1, -1) / num_hiddens)
        self.P[0, :, 0::2] = torch.sin(X) # P的偶数列填入sin(X)
        try:
            self.P[0, :, 1::2] = torch.cos(X) # P的奇数列填入cos(X). 当最后一列idx=num_hiddens-1是奇数时,正好填入
        except:
            self.P[0, :, 1::2] = torch.cos(X[:, :-1]) # 当最后一列idx=num_hiddens-1是偶数时,去掉X的最后一列再填入
    
    def forward(self, X):
        assert X.shape[-1] == self.P.shape[-1], "input's feature dim not equal to num_hiddens arg of TrigonoPosEnc"
        assert X.shape[1] <= self.P.shape[1], "input's sequence length exceeds max_len arg of TrigonoPosEnc"
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class LearnPosEnc(nn.Module):
    '''
    args: seq_len, num_hiddens, dropout
        seq_len: the sequence length of input X
        num_hiddens: the feature dimentions of input X, simply as d
        dropout: dropout rate. Regularization on final X + PosEnc(X)
    
    inputs: X
        X's shape: (batch_size, seq_len, d=num_hiddens)
    
    returns: denoted as S (in-place, same device with input X and net)
        S's shape: (batch_size, seq_len, d=num_hiddens)

    explains:
        THe Learnable positional encodes P shall be shared in different position encoding layers
        When input X (batch_size, seq_len, d) arrives, Learnable positional encoding P with shape (seq_len, d)
        is added to every sample of X. P with shape (seq_len, d), whose elements are all learnable parameters.
    '''
    def __init__(self, seq_len, num_hiddens, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # create the PosEncodes parameters with the same shape as input X's each sample
        self.P = nn.Parameter(torch.randn(1, seq_len, num_hiddens))

    def forward(self, X):
        assert X.shape[-1] == self.P.shape[-1], "input's feature dim not equal to num_hiddens arg of LearnPosEnc"
        assert X.shape[1] == self.P.shape[1], "input's sequence length not equal to seq_len arg of LearnPosEnc"
        X = X + self.P
        return self.dropout(X)