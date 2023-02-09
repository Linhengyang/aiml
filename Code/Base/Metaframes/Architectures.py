import torch
from torch import nn

########## Basic Encoder-Decoder architectures ##########
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture.

    inputs: source input data batch, others(optional)

    return: denoted as enc_outputs
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, enc_X, *args, **kwargs):
        raise NotImplementedError

class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture.

    inputs: 
        target input data batch
        state from init_state method

    methods:
        init_state: enc_outputs as input, return state
    
    return: denoted as dec_outputs
    """
    def __init__(self):
        super().__init__()

    def init_state(self, enc_outputs, *args, **kwargs):
        raise NotImplementedError

    def forward(self, dec_X, *args, **kwargs):
        raise NotImplementedError

########## Attention-Customized decoder ##########
class AttentionDecoder(Decoder):
    """The base attention-based decoder interface.
    inputs: 
        1. target input data batch
        2. state from init_state method

    methods:
        init_state: enc_outputs as input, return state
    
    attributes:
        attention_weights

    return: denoted as dec_outputs
    """
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture.

    args: encoder, decoder

    inputs: enc_X, dec_X, others(optional)
        encoder(enc_X, others) --> enc_outputs
        extract enc_outputs --> state
        decoder(dec_X, state) --> dec_outputs
    
    returns: denoted as dec_outputs, y_hat is included

    explains: be careful to design proper work for infer and train mode.
    """
    def __init__(self):
        super().__init__()

    def forward(self, enc_X, dec_X, *args, **kwargs):
        raise NotImplementedError
