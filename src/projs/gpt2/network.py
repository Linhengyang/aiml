from ...core.nn_components.meta_frames import Decoder, DecoderOnly
from ...core.nn_components.root_layers.position_encoding import LearnAbsPosEnc
from ...core.nn_components.sub_modules._gpt2 import GPT2DecoderBlock
import torch.nn as nn
import math
import torch





class gpt2(DecoderOnly):
    pass