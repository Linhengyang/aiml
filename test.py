# test.py
import torch
import typing as t
import os
import math
import json

temp = '../cache/temp/'
gpt2_resource_dir = '../../resource/llm/gpt/gpt2/'


from src.core.utils.text.tokenizer import boostBBPETokenizer, ENDOFTEXT
from src.core.utils.common.seq_operation import pack_seq_to_batch_slide


if __name__ == "__main__":
    pass