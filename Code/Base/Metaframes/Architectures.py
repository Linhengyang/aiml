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
    
    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture.

    inputs: target input data batch, state extracted from enc_outputs bt init_state method

    return: denoted as dec_outputs
    """
    def __init__(self):
        super().__init__()

    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
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
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        return self.decoder(dec_X, dec_state)

########## Attention-Customized decoder ##########
class AttentionDecoder(Decoder):
    """The base attention-based decoder interface.
    args: encoder, decoder

    inputs: enc_X, dec_X, others(optional)
        encoder(enc_X, others) --> enc_outputs
        extract enc_outputs --> state
        decoder(dec_X, state) --> dec_outputs
    
    returns: denoted as dec_outputs, y_hat is included

    attributes: attention_weights
    
    explains: be careful to design proper work for infer and train mode.
    """
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError
