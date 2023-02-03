import torch
from torch import nn
from ..Base.SubModules.AddLNorm import AddLNorm
from ..Base.RootLayers.AttentionPools import MultiHeadAttention

class PositionWiseFFN(nn.Module):
    '''
    args: ffn_num_hiddens, ffn_num_outputs
        ffn_num_hiddens: the hidden size inside the MLP
        ffn_num_outputs: output size of the MLP, usually the same as the input size
    
    inputs: X

    returns: denoted as O
    
    explains:
        Perform the same MLP on every position. So only one MLP is enough.
    '''
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)
    
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class TransformerEncoderBlock(nn.Module):
    '''
    components:
        1. multihead attention(self-att)
        2. addLnorm
        3. positionwiseFFN
        4. addLnorm
    
    inputs: enc_X, valid_lens(optional)
        enc_X's shape: (batch_size, seq_len, d_dim)
        valid_lens(optional)'s shape: (batch_size,) since it's self-att here
    
    returns: denoted as enc_O
        enc_O's shape: (batch_size, seq_len, d_dim), the same as enc_X
    
    explains:
        keep batch shape at every layer's input/output through the block
    '''
    def __init__(self, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, num_hiddens, dropout, use_bias)
        self.addlnorm1 = AddLNorm(num_hiddens, dropout)
        self.PosFFN = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addlnorm2 = AddLNorm(num_hiddens, dropout)
    
    def forward(self, X, valid_lens):
        self_att_X = self.attention(X, X, X, valid_lens)
        Y = self.addlnorm1(X, self_att_X)
        posffn_Y = self.PosFFN(Y)
        return self.addlnorm2(Y, posffn_Y)
