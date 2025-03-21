import math
import torch
from torch import nn

class TrigonoAbsPosEnc(nn.Module):
    '''
    args:
        num_hiddens: the feature dimentions of input X, simply as d
    
    inputs:
        position_ids: 1D tensors of int64, shall be inside [0, max_len-1]
    
    returns: (1, len(position_ids), num_hiddens)int64 position embedding of position_ids

    explains:
        Given input position_ids(int64), TrigonoAbsPosEnc returns embeddings of position_ids with
            for k in position_ids,
                (k, 2j) element: sin( k/10000^(2j/d) )
                (k, 2j+1) element is cos( k/10000^(2j/d) )
    '''
    def __init__(self, num_hiddens, max_len=1000):
        super().__init__()
        self.P = torch.zeros((1, max_len, num_hiddens), device=self.device) # create a long enough P with same d(d_dim) as input

        # X 2-D tensor, 行idx是0到max_len-1, 列idx是0到num_hiddens-1之内的偶数
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32).reshape(1, -1) / num_hiddens)
        
        self.P[0, :, 0::2] = torch.sin(X) # P的偶数列填入sin(X)
        try:
            self.P[0, :, 1::2] = torch.cos(X) # P的奇数列填入cos(X). 当最后一列idx=num_hiddens-1是奇数时,正好填入
        except:
            self.P[0, :, 1::2] = torch.cos(X[:, :-1]) # 当最后一列idx=num_hiddens-1是偶数时,去掉X的最后一列再填入
    

    def forward(self, position_ids):
        # position_ids: 1D tensors of int64, shall be inside [0, max_len-1]
        
        return self.P[:, position_ids, :] # shape as (1, len(position_ids), num_hiddens)






class LearnAbsPosEnc(nn.Module):
    '''
    args:
        seq_len: the sequence length
        num_hiddens: the feature dimentions of input X, simply as d
    
    inputs:
        position_ids: 1D tensors of int64, shall be inside [0, seq_len]. 0 stands for <bos>
    
    returns: (1, len(position_ids), num_hiddens)learnable position embedding of position_ids

    explains:
        The Learnable absolute positional encodes P shall be shared in different position encoding layers.

        PosEnc with shape (1, seq_len+1, num_hiddens), whose elements are all learnable parameters. the +1 learnable is for <bos> position

        Given input position_ids(int64), selected learnable positional encoding PosEnc with shape 
        (1, len(position_ids), num_hiddens) shall be added to every corresponding steps of data sample
    '''
    def __init__(self, seq_len, num_hiddens):
        super().__init__()
        self.register_parameter('PosEnc', nn.Parameter(torch.randn(1, seq_len+1, num_hiddens)))

    def forward(self, position_ids):
        # position_ids: 1D tensors of int64, shall be inside [0, seq_len]
        
        return self.PosEnc[:, position_ids, :] # shape as (1, len(position_ids), num_hiddens)
