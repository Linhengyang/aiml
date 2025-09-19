import torch
import math
from typing import Tuple
from torch import nn
from ..root_layers.attention_pool import CasualMHA


class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GeLUFFN(nn.Module):
    '''
    GPT 经典 FeedForward 实现(with GeLU)

    [..., D] --linear--> [..., 4*D] --non_linear_act(gelu/relu)--> [..., 4D] --linear--> [..., D] --dropout--> [..., D]
    '''
    def __init__(self, embd_size, use_bias, resid_p_drop):
        super().__init__()
        self.W_in = nn.Linear(embd_size, 4*embd_size, use_bias)
        self.gelu = GeLU()
        self.W_out = nn.Linear(4*embd_size, embd_size, use_bias)
        self.drop = nn.Dropout(resid_p_drop)

    def forward(self, x):
        return self.drop(self.W_out(self.gelu(self.W_in(x))))



class GPT2DecoderBlock(nn.Module):
    '''
    pre-layer_normalization 的 decoder block 架构图:
    
        -----------------add---------------->|       ------------add------------>|
      x --layer_norm-->|--casual_attention-->|--> x_ --layer_norm-->--gelu_ffn-->|--> y
    kv_cache(if any)-->|                      --> new_kv_cache(if need)
   attn_mask(if any)-->|
   positions(if any)-->|

    x / x_ / y have same shape
    '''
    def __init__(self,
                 embd_size:int,
                 num_head:int,
                 use_bias:bool,
                 max_context_size:int,
                 attn_p_drop:float,
                 resid_p_drop:float,
                 use_cached_casual_mask:bool,
                 use_rope:bool
                 ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embd_size)
        self.casual_attention = CasualMHA(embd_size, num_head, use_bias, max_context_size, attn_p_drop,
                                          resid_p_drop, use_cached_casual_mask, use_rope)
        self.layer_norm2 = nn.LayerNorm(embd_size)
        self.gelu_ffn = GeLUFFN(embd_size, use_bias, resid_p_drop)

    def forward(self,
                x:torch.Tensor,
                kv_cache:Tuple[torch.Tensor, torch.Tensor]|None=None,
                return_cache:bool = False,
                attention_mask:torch.Tensor|None = None,
                positions:torch.Tensor|None = None):
        
        attn_result, new_kv_cache = self.casual_attention(self.layer_norm1(x), kv_cache, return_cache, attention_mask, positions)
        x_ = x + attn_result
        y = x_ + self.gelu_ffn(self.layer_norm2(x_))
        
        return y, new_kv_cache