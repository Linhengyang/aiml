from ...core.nn_components.meta_frames import Encoder, AttentionDecoder, EncoderDecoder
from ...core.nn_components.root_layers.position_encoding import TrigonoAbsPosEnc
from ...core.nn_components.sub_modules._gpt2 import GPT2DecoderBlock
import torch.nn as nn
import math
import torch





class gpt2(AttentionDecoder):
    pass