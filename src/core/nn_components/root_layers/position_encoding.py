import math
import torch
from torch import nn
import typing as t

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
        PosEnc = torch.zeros((max_len, num_hiddens))
        # X 2-D tensor, 行idx是0到max_len-1, 列idx是0到num_hiddens-1之内的偶数
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32).reshape(1, -1) / num_hiddens)
        
        PosEnc[:, 0::2] = torch.sin(X) # P的偶数列填入sin(X)
        try:
            PosEnc[:, 1::2] = torch.cos(X) # P的奇数列填入cos(X). 当最后一列idx=num_hiddens-1是奇数时,正好填入
        except:
            PosEnc[:, 1::2] = torch.cos(X[:, :-1]) # 当最后一列idx=num_hiddens-1是偶数时,去掉X的最后一列再填入

        self.register_buffer('PosEnc', PosEnc)

    def forward(self, position_ids: torch.Tensor):
        '''
        position_ids: 2D tensors of int64 as of (batch_size, num_positions) / 1D tensor of int64 as (num_positions,)
        若 position_ids 是 1D tensor, 那么 input tokens-embedding + positions-embedding 的过程依赖 broadcast
        values 是 position index starting from 0. shall be inside [0, max_len-1]
        '''

        return self.PosEnc[position_ids, :] # shape as (batch_size, len(position_ids), num_hiddens)






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
        self.register_parameter('PosEnc', nn.Parameter(torch.randn(max_possible_posNum, num_hiddens)))

    def forward(self, position_ids: torch.Tensor):
        '''
        position_ids: 2D tensor of int64 as of (batch_size, num_positions) / 1D tensor of int64 as (num_positions,)
        若 position_ids 是 1D tensor, 那么 input tokens-embedding + positions-embedding 的过程依赖 broadcast
        values 是 position index starting from 0. shall be inside [0, max_possible_posNum-1]
        '''
        
        return self.PosEnc[position_ids, :] # shape as (batch_size, len(position_ids), num_hiddens) / (len(position_ids), num_hiddens)




from dataclasses import dataclass

@dataclass
class RoPEConfig:
    dim: int
    base: float = 10000.0     # 频率基数
    rope_scale: float = 1.0   # 角频率缩放(>1.0 可"拉长"上下文)



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
    与其他PE直接加在embedding上不同, RoPE是作用在q/k上的(attention计算之前): 
        attention计算中,q/k在last dim计算内积. 若将q/k last dim上的每1对(2个)维度构成一个复平面, 对每个复平面上的二维向量作旋转, 
        旋转的角度和绝对位置相关, 这样在qk计算内积时,内积与相对距离(绝对位置之差)有关, 而不是绝对位置.
    在实际RoPE过程中, 视q/k的每一个head作为两两维度旋转的总维度, 即帮助每个head内部抓住相对位置信息.

    其他PE是“从embedding矩阵中抽取位置代表的vector”, 所以预先对总位置数量有预设; RoPE是根据位置index确定q/k不同的旋转方式, 故不需要预设总位置数量。
    所以RoPE对位置总量的延展性也更好。

    theta向量: 由绝对位置确定（绝对位置 结合 周期频率信号）
    cos theta向量: 用于构建旋转矩阵的 左上角 / 右下角, 用于“偶数维度”贡献“新偶数维度”的比例，和“奇数维度”贡献“新奇数维度”的比例
    sin theta向量: 用于构建旋转矩阵的 左下角 / 右上角, 用于“奇数维度”贡献“新偶数维度”的比例（负号），和“偶数维度”贡献“新奇数维度”的比例
    even_dims偶数维度向量: 从index 0维开始的偶数维度分量
    odd_dims 奇数维度向量: 从index 1维开始的奇数维度分量
        cos_theta * even_dims + sin_theta * (-odd_dims) -> rotated even_dims
        cos_theta * odd_dims + sin_theta * even_dims -> rotated odd_dims
        even_dims, odd_dims 就是 original tensor, -odd_dims, even_dims 就是 rotate_half_on_last_dim(original tensor)
    综上：
        rotated tensor = cos_theta * original_tensor + sin_theta * rotate_half_on_last_dim(original_tensor)
    这里 cos_theta / sin_theta 的last dim size = D/2, original_tensor 的last dim size = D
    cos_theta需要先在 last dim 作interleave的duplicate, 延展成last dim size = D 后, 再作运算, 可得 last dim size = D 的 rotated tensor

    主流实现中 q/k 的shape是(B,H,S,D), 即(batch_size, num_heads, seq_length, dim_per_head). 不过偶尔也会有(B,S,H,D)的实现.
    cos_theta / sin_theta 本质上只需要 S 和 D 维度上的信息即可, 其余两维可靠广播. 实际计算过程：

    position_ids: (batch_size, num_positions=seq_length) --> theta(angle) matrix: (batch_size, seq_length, dim_per_head/2)
    --unqueeze在num_heads维度--> theta tensor: (batch_size, 1, seq_length, dim_per_head/2) -->
    cos/sin tensors: (batch_size, 1, seq_length, dim_per_head/2) --interleaved_duplicate--> (batch_size, 1, seq_length, dim_per_head)
    q/k tensors: (batch_size, num_heads, seq_length, dim_per_head)
    rotate_half_on_last_dim(q/k) tensors --> (batch_size, num_heads, seq_length, dim_per_head)

    cos * q + sin * rorate_half_on_last_dim(q) = q_rotate
    '''
    def __init__(self, config: RoPEConfig):
        super().__init__()
        assert config.dim % 2 == 0, f'RoPE dim must be even'
        self.config = config

        # 仿照绝对位置编码 TrigonoAbsPosEnc 构造绝对位置相关的周期频率 w_i = base ^ (-2i/d)
        # 向量化 2i vector, dtype 设定为 float, 在 float 中计算 --> inv_freq shape as (dim_per_head/2, )
        inv_freq = 1.0 / (config.base ** (torch.arange(0, config.dim, 2).float() / config.dim)) # float32
        # 频率缩放(rope_scale>1相当于减小角度增速，延展上下文，即数值上提高上下文之间的相关程度)
        inv_freq = inv_freq / config.rope_scale
        # 注册成常量 buffer inv_freq: (dim_per_head/2, )
        self.register_buffer("inv_freq", inv_freq, persistent=False)


    @torch.no_grad()
    def _angles(self, position_ids: torch.Tensor) -> torch.Tensor:
        '''
        position_ids: shape(batch_size, num_positions), dtype int64, range >= 0
        inv_freq: shape(dim/2,), dtype float

        theta tensor: theta = 绝对位置 index p 乘以 inv_frequence 10000^-(2i/d). 这里 2i 是 0 -> dim 的vector

        return:
        theta tensor: (batch_size, seq_length, dim_per_head/2)
        
        position_ids 的每一个值 p 乘以 inv_freq 频率信号, 即为 每一个位置在 dim 维上的频率信号
        '''
        pos = position_ids.to(dtype=torch.float32)
        # 爱因斯坦求和：广播乘法的一种表示，指定 pos shape (b,s), inv_freq shape (d,), 指定广播乘法结果 shape (b,s,d)
        # 每一个position value 乘以 inv_freq --> 得到 (dim/2,) 的1D tensor --> (batch_size, num_positions, dim/2)
        # 等价于 pos[:, :, None] * inv_freq[None, None, :]
        return torch.einsum("bs,d->bsd", pos, self.inv_freq) # float32


    def get_sin_cos(self, position_ids: torch.Tensor, device=None, dtype=None, broadcast_axis=1) -> t.Tuple[torch.Tensor, torch.Tensor]:
        '''
        根据 theta tensor(batch_size, seq_length, dim_per_head/2), 得出可供旋转计算的 cos tensor 和 sin tensor
        cos tensor 和 sin tensor 的 shape 根据 q/k tensor的形状确定:
            q/k形状[B, H, S, D], 则 cos/sin 形状是 (batch_size, 1, seq_length, dim_per_head/2)
            q/k形状[B, S, H, D], 则 cos/sin 形状是 (batch_size, seq_length, 1, dim_per_head/2)
            q/k形状[B, S, D], 则 cos/sin 形状是 (batch_size, seq_length, dim_per_head/2)
        cos/sin tensor 的 dtype 是 torch.float32
        '''
        theta = self._angles(position_ids) # (batch_size, seq_length, dim_per_head/2), float32
        if dtype is None:
            dtype = theta.dtype # float32
        if device is None:
            device = theta.device
        
        # 先算半维度的 cos/sin
        cos_half = torch.cos(theta).to(dtype=dtype, device=device) # (batch_size, seq_length, dim_per_head/2)
        sin_half = torch.sin(theta).to(dtype=dtype, device=device) # (batch_size, seq_length, dim_per_head/2)

        # 处理 head 维度, 方便与 q/k 计算时的广播
        if not isinstance(broadcast_axis, int):
            # (batch_size, seq_length, dim_per_head/2)
            pass
        else:
            # (batch_size, seq_length, dim_per_head/2) -> 
            #       (batch_size, 1, seq_length, dim_per_head/2) 或者 (batch_size, seq_length, 1, dim_per_head/2)
            cos_half = cos_half.unsqueeze(broadcast_axis)
            sin_half = sin_half.unsqueeze(broadcast_axis)
        
        # 在最后一维交叉复制偶/奇相邻两个维度
        # (..., dim_per_head/2) -> (..., dim_per_head)
        cos = cos_half.repeat_interleave(2, dim=-1)
        sin = sin_half.repeat_interleave(2, dim=-1)

        # shape 有三种可能
        # (B, 1, seq_len, d)
        # (B, seq_len, 1, d)
        # (B, seq_len, d), no HEAD dim
        return cos, sin


    def apply_rope(self, x:torch.Tensor, cos:torch.Tensor, sin:torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        '''
        将 RoPE 作用在 x 上, 返回 x'
        x shape:                    (B, H, seq_len, d) / (B, seq_len, H, d) / (B, seq_len, D)
        对应 broadcast_axis 分别是           1                   2                  None

        cos/sin shape:              (B, 1, seq_len, d) / (B, seq_len, 1, d) / (B, seq_len, d)
        是和 x 的 position_ids 和 broadcast_axis 相对应的 cos 和 sin
        '''
        # 记录下 x 原来的 dtype
        dtype = x.dtype

        # 使用 float32 格式计算 RoPE, 使得整个计算过程保持稳定.
        xf = x.to(torch.float32)
        x_rotate = cos * xf + sin * rotate_half_on_last_dim(xf)

        return x_rotate.to(dtype)