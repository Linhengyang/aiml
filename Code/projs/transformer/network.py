from ...Base.MetaFrames import Encoder, Decoder, AttentionDecoder, EncoderDecoder

class TransformerEncoder(Encoder):

    def __init__(self):
        super().__init__()
    
    def forward(self, src_X, *args):
        raise NotImplementedError

class TransformerDecoder(AttentionDecoder):

    def __init__(self):
        super().__init__()
    
    def init_state(self):
        raise NotImplementedError

    def forward(self, src_X, *args):
        raise NotImplementedError

class Transformer(EncoderDecoder):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self):
        raise NotImplementedError