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




from dataclasses import dataclass

@dataclass
class RoPEConfig:
    dim: int
    base: float = 10000.0     # 频率基数
    rope_scale: float = 1.0   # 角频率缩放(>1.0 可"拉长"上下文)
    interleaved: bool = True  # 是否使用 交织布局(即每相邻偶奇两位作二维平面)



def rotate_half_on_last_dim(x: torch.Tensor) -> torch.Tensor:
    '''
    把tensor x 的最后一维按(even, odd)成对视作二维平面上的向量(x, y), 返回逆时针旋转90度的正交向量(-y, x)
    '''
    even_dims, odd_dims = x[..., ::2], x[..., 1::2]
    # even_dims: (..., 3) where last dim is 0/2/4 from original
    # odd_dims: (..., 3) where last dim is 1/3/5 from original
    # stack at dim -1 --> shape as (..., 3, 2) where 3 of last dim is neg1/0, neg3/2, neg5/4 --> reshape to input x
    return torch.stack((-odd_dims, even_dims), dim=-1).reshape_as(x)



class RotaryPosEnc(nn.Module):
    '''
    与其他PE直接加在embedding上不同, RoPE是作用在q/k上的: q/k上的每1对(2个)维度构成一个复平面, 对每个复平面上的二维向量作旋转, 旋转的角度和绝对位置相关
    这样在qk计算时,旋转后的qk内积与相对距离(绝对位置之差)有关, 而不是绝对位置.
    theta向量: 由绝对位置确定（绝对位置 结合 周期频率信号）
    cosine theta向量: 用于构建旋转矩阵的 左上角 / 右下角, 用于“偶数维度”贡献“新偶数维度”的比例，和“奇数维度”贡献“新奇数维度”的比例
    sine theta向量: 用于构建旋转矩阵的 左下角 / 右上角, 用于“奇数维度”贡献“新偶数维度”的比例（负号），和“偶数维度”贡献“新奇数维度”的比例
    even_dims偶数维度向量: 从index 0维开始的偶数维度分量
    odd_dims奇数维度向量: 从index 1(若有)维开始的奇数维度分量
        cosine_theta @ even_dims + sine_theta @ (-odd_dims) -> rotated even_dims
        cosine_theta @ odd_dims + sine_theta @ even_dims -> rotated odd_dims
        even_dims, odd_dims 就是 original tensor, -odd_dims, even_dims 就是 rotate_half_on_last_dim(original tensor)
        rotated tensor = cosine_theta * original_tensor + sine_theta * rotate_half_on_last_dim(original_tensor)
    '''
    def __init__(self, config: RoPEConfig):
        super().__init__()
        assert config.dim % 2 == 0, f'RoPE dim must be even'
        self.config = config

        # 仿照绝对位置编码 TrigonoAbsPosEnc 构造绝对位置相关的周期频率 w_i = base ^ (-2i/d)
        inv_freq = 1.0 / (config.base ** ())