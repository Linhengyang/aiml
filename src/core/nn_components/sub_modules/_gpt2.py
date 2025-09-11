import torch
from torch import nn
from ..root_layers.layer_normalization import AddLayerNorm
from ..root_layers.attention_pool import MultiHeadAttention
from ..root_layers.feedforward import PositionWiseFFN





class GPT2DecoderBlock(nn.Module):
    '''
    '''
    def __init__(self):
        super().__init__()

    def forward(self):
        pass