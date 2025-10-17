# test.py
import torch
import typing as t
import os
import math
import json

temp = '../cache/temp/'
gpt2_resource_dir = '../../resource/llm/gpt/gpt2/'



from src.kits.tokenizer_kit.gpt import gpt2Tokenizer


if __name__ == "__main__":
    # 测试 tokenizer
    tokenizer_path = os.path.join(gpt2_resource_dir, 'tokenizer.json')

    tok = gpt2Tokenizer()
    tok.from_doc(tokenizer_path)

    text = '! hello I am linhengyang<|endoftext|>'
    encoded = tok.encode(text, allowed_special='all')
    text_ = tok.decode(encoded)
    print(encoded)
    assert text_ == text

    # 测试加载 模型(pure-torch)
    