from ...Base.MetaFrames import Encoder, Decoder, AttentionDecoder, EncoderDecoder
from ...Base.RootLayers.PositionalEncodings import TrigonoAbsPosEnc, LearnAbsPosEnc
from ...Modules._transformer import TransformerEncoderBlock, TransformerDecoderBlock
import torch.nn as nn
import math

class TransformerEncoder(Encoder):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

class TransformerDecoder(AttentionDecoder):
    def __init__(self):
        super().__init__()
        pass

    def init_state(self):
        pass

    def forward(self):
        pass

class Transformer(EncoderDecoder):
    ## 整体两种实现模式:
    # 1是encoder和decoder一起训练, 但分开使用, 这样transformer本身只需考虑train batch input的场景, infer的循环放在pred函数里
    # 2是encoder和decoder一起训练, 一起使用, 这样transformer要注意infer sample input的场景, infer的循环放在forward里
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src_X, tgt_X, src_valid_lens):
        pass
