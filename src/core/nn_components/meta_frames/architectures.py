import torch
from torch import nn

########## Basic Encoder/Decoder architectures ##########

########## Encoder ##########
class Encoder(nn.Module):
    """The basic encoder interface"""
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def attention_weights(self):
        raise NotImplementedError


########## Decoder ##########
class Decoder(nn.Module):
    """The basic decoder interface """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    @property
    def attention_weights(self):
        raise NotImplementedError


########## Encoder-Decoder ##########
class EncoderDecoder(nn.Module):
    """The basic class for encoder-decoder architecture"""
    def __init__(self, encoder:Encoder, decoder:Decoder):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


########## Decoder-Only ##########
class DecoderOnly(nn.Module):
    """The basic class for decoder-only architecture"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    @torch.no_grad()
    def generate(self, *args, **kwargs):
        raise NotImplementedError