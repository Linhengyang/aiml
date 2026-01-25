import torch
import math
from typing import Tuple
from torch import nn
from src.core.layers.feedforward import relu_ffn



class BERTEncoderBlock(nn.Module):
    '''
    post-layer_normalization 的 encoder block 架构图:
    
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
                 ffn_hidden_size:int,
                 attn_p_drop:float,
                 resid_p_drop:float
                 ):
        super().__init__()


    def forward(self,
                x:torch.Tensor,
                attention_mask:torch.Tensor|None = None):
        
        return
