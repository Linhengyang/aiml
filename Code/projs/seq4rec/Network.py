from ...Base.MetaFrames import Encoder, Decoder, AttentionDecoder, EncoderDecoder
from ...Base.RootLayers.PositionalEncodings import TrigonoAbsPosEnc, LearnAbsPosEnc
from ...Base.SubModules.AddLNorm import AddLNorm
import torch.nn as nn

class seq4recEncoder(nn.Module):
    def __init__(self, num_hiddens, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = nn.LazyLinear(num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(num_hiddens)
        self.addlnorm = AddLNorm(num_hiddens, dropout)
        self.tanh = nn.Tanh()
        self.head = nn.LazyLinear(2)


    def forward(self, X):
        Y = self.relu( self.dense1(X) )
        f_Y = self.dense2(Y)
        return self.head( self.tanh( self.addlnorm(Y, f_Y) ) )
    

