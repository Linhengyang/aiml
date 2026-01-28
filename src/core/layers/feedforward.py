import torch
import math
from torch import nn


class relu_ffn(nn.Module):
    '''
    transformer 经典 FeedForward 实现(with ReLU), 即 Position-Wise FeedForward Neural Network
    '''
    def __init__(self, output_size, ffn_hidden_size, resid_p_drop):
        super().__init__()
        self.W_in = nn.Linear(output_size, ffn_hidden_size)
        self.relu = nn.ReLU()
        self.W_out = nn.Linear(ffn_hidden_size, output_size)
        self.drop = nn.Dropout(resid_p_drop)
    
    def forward(self, x):
        return self.drop(self.W_out(self.relu(self.W_in(x))))



class gelu_ffn(nn.Module):
    '''
    GPT 经典 FeedForward 实现(with GeLU):
    [..., D] --linear--> [..., 4*D] --non_linear_act(gelu/relu)--> [..., 4D] --linear--> [..., D] --dropout--> [..., D]
    '''
    def __init__(self, embd_size, use_bias, resid_p_drop):
        super().__init__()
        self.W_in = nn.Linear(embd_size, 4*embd_size, use_bias)
        self.gelu = nn.GELU()
        self.W_out = nn.Linear(4*embd_size, embd_size, use_bias)
        self.drop = nn.Dropout(resid_p_drop)

    def forward(self, x):
        return self.drop(self.W_out(self.gelu(self.W_in(x))))