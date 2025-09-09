# test.py
import sys
import typing as t
import torch

cache_play = '../cache/playground/'


# from src.core.utils.text.tokenizer import asyncBBPETokenizer
import pyarrow as pa
from src.core.nn_components.root_layers.position_encoding import RotaryPosEnc, RoPEConfig

if __name__ == "__main__":
    rope_config = RoPEConfig(dim=4)
    rope = RotaryPosEnc(rope_config)
    position_ids = torch.tensor([0,1,2,3]).unsqueeze(0)
    cos1, sin1 = rope.get_sin_cos(position_ids)

    position_id = torch.tensor([2]).unsqueeze(0)
    cos2, sin2 = rope.get_sin_cos(position_id)

    print("cos1: ", cos1)
    print("cos2: ", cos2)