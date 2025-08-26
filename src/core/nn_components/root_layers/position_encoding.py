import math
import torch
from torch import nn

class TrigonoAbsPosEnc(nn.Module):
    '''
    args:
        num_hiddens: the feature dimentions for Position ID to embed, simply as d
    
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

        # PosEnc: a long enough matrix with same d(d_dim) as input, (max_len, d_dim)
        PosEnc = torch.zeros((1, max_len, num_hiddens))
        # X 2-D tensor, 行idx是0到max_len-1, 列idx是0到num_hiddens-1之内的偶数
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32).reshape(1, -1) / num_hiddens)
        
        PosEnc[0, :, 0::2] = torch.sin(X) # P的偶数列填入sin(X)
        try:
            PosEnc[0, :, 1::2] = torch.cos(X) # P的奇数列填入cos(X). 当最后一列idx=num_hiddens-1是奇数时,正好填入
        except:
            PosEnc[0, :, 1::2] = torch.cos(X[:, :-1]) # 当最后一列idx=num_hiddens-1是偶数时,去掉X的最后一列再填入

        self.register_buffer('PosEnc', PosEnc)

    def forward(self, position_ids):
        # position_ids: 1D tensors of int64, shall be inside [0, max_len-1]
        
        return self.PosEnc[:, position_ids, :] # shape as (1, len(position_ids), num_hiddens)






class LearnAbsPosEnc(nn.Module):
    '''
    args:
        max_possible_posNum: how many possible 1D positions are there in total
            e.g, if there are positions of 1,2,3,4,5 without 0, then max_possible_posNum = 5
            e.g, if there are positioins of 1,2,3,4,5 but position 0 still exists, then max_possible_posNum = 6

        For sequence data, max_possible_posNum can be 1 + sequence length if BOS should be considered

        Given max_possible_posNum, the layer creates positions by using [0, 1, ..., max_possible_posNum-1] to index

        num_hiddens: the feature dimentions for Position ID to embed, simply as d
    
    inputs:
        position_ids: 1D tensors of int64, shall be inside [0, max_possible_posNum-1]
    
    returns: (1, len(position_ids), num_hiddens) learnable position embedding of position_ids

    explains:
        The Learnable absolute positional encodes P shall be shared in different position encoding layers.

        PosEnc with shape (1, max_possible_posNum, num_hiddens), whose elements are all learnable parameters.

        Given input position_ids(int64), selected learnable positional encoding PosEnc with shape 
        (1, len(position_ids), num_hiddens) shall be added to every corresponding steps of data sample
    '''
    def __init__(self, max_possible_posNum, num_hiddens):
        super().__init__()
        self.register_parameter('PosEnc', nn.Parameter(torch.randn(1, max_possible_posNum, num_hiddens)))

    def forward(self, position_ids):
        # position_ids: 1D tensors of int64, shall be inside [0, max_possible_posNum-1]
        
        return self.PosEnc[:, position_ids, :] # shape as (1, len(position_ids), num_hiddens)






class RotaryPosEnc(nn.Module):
    '''
    与其他PE直接加在embedding上不同, RoPE是作用在q/k上的: q/k上的每1对(2个)维度构成一个复平面, 对每个复平面上的二维向量作旋转, 旋转的角度和绝对位置相关
    这样在qk计算时,旋转后的qk内积与相对距离(绝对位置之差)有关, 而不是绝对位置.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)