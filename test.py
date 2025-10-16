# test.py
import torch
import typing as t
import os
import math
import json

temp = '../cache/temp/'
gpt2_resource_dir = '../../resource/llm/gpt/gpt2/'


from src.core.utils.text.tokenizer import boostBBPETokenizer, ENDOFTEXT
from src.kits.tokenizer_kit.gpt_style import gpt2Tokenizer


if __name__ == "__main__":
    tokenizer_path = os.path.join(gpt2_resource_dir, 'tokenizer.json')
    print(tokenizer_path)
    with open(tokenizer_path) as f:
        raw_tok = json.load(f)

    print(len(raw_tok['model']['vocab']))
