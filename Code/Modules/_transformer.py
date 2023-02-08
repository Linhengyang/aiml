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
        encode source sequence time 1 to T directly to deep sequence time 1 to T, that is:
            f(time 1 to T) --> node 1 to T on next layer
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

class TransformerDecoderBlock(nn.Module):
    '''
    components:
        1. masked multihead attention(self-att but using a mask to be auto-regressive)
        2. addLnorm
        3. encoder-decoder attention(use sequences from decoder as queries, keys&values are from encoder)
        4. addLnorm
        5. positionwiseFFN
        6. addLnorm
    
    inputs: dec_X, state
        dec_X's shape: (batch_size, seq_len, d_dim)
        state:
            self-att层, 需要一些额外的信息, 满足infer过程的需要
            enc-dec-att层, 需要在这里输入encoder output相关信息, 即decoder在这里从source sequence中吸取信息
    
    returns: denoted as dec_O
        dec_O's shape: (batch_size, seq_len, d_dim), the same as dec_X
    
    explains:
        keep batch shape at every layer's input/output through the block
    '''
    def __init__(self, blk_ind, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias=False):
        super().__init__()
        self.blk_ind = str(blk_ind)
        self.attention1 = MultiHeadAttention(num_heads, num_hiddens, dropout, use_bias)
        self.addlnorm1 = AddLNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(num_heads, num_hiddens, dropout, use_bias)
        self.addlnorm2 = AddLNorm(num_hiddens, dropout)
        self.PosFFN = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addlnorm3 = AddLNorm(num_hiddens, dropout)
    
    def forward(self, X, enc_info, infer_recorder=None):
        enc_output, enc_valid_lens = enc_info
        # X' shape, train: (batch_size, seq_len, d_dim), infer: (1, 1, d_dim)
        # enc_output'shape, train: (batch_size, seq_len, d_dim), infer: (1, seq_len, d_dim)
        # enc_valid_lens's shape, train: (batch_size,), infer: (1,) or None
        if self.training: # train过程中
            assert X.shape == enc_output.shape, 'training: encoder output & decoder input block ' + self.blk_ind + ' are not in same shape'
            assert enc_output.shape[0] == enc_valid_lens.shape[0], 'encoder output & encoder valid lens have different batch_size'
            batch_size, seq_len = X.shape[:-1]
            mask = torch.arange(1, seq_len+1, dtype=torch.int32, device=X.device).repeat(batch_size, 1)
            KVs = X # 自注意力, 用上面的mask实现auto-regressive, train过程中不需要infer_recorder
        else: # infer过程中
            # infer_recoder是一个list of (1, _, d_dim) tensor
            assert type(infer_recorder) == dict, 'in infer mode, a dictionary as infer recorder should be input'
            try: # 从<bos> infer第一个token时, infer_recorder中存在空的元素. 
                infer_recorder[self.blk_ind] = torch.cat(infer_recorder[self.blk_ind], X, dim=1)
            except KeyError:
                infer_recorder[self.blk_ind] = X
            KVs = infer_recorder[self.blk_ind] # 用所有recorded tokens作自注意力
            mask = None # infer过程中不需要mask
        X2 = self.attention1(X, KVs, KVs, mask)
        Y = self.addlnorm1(X, X2)
        Y2 = self.attention2(Y, enc_output, enc_output, enc_valid_lens)
        Z = self.addlnorm2(Y, Y2)
        Z2 = self.PosFFN(Z)
        return self.addlnorm3(Z, Z2), infer_recorder
