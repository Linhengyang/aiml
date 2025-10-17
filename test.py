# test.py
import torch
import typing as t
import os
import math
import json

temp = '../cache/temp/'
gpt2_resource_dir = '../../resource/llm/gpt/gpt2/'



from workspace.aiml.src.kits.huggingface.tokenizer_adapt import gpt2Tokenizer
from src.projs.gpt2.network import gpt2, gpt2Config


if __name__ == "__main__":
    # 测试 tokenizer
    tokenizer_path = os.path.join(gpt2_resource_dir, 'tokenizer.json')

    tok = gpt2Tokenizer()
    tok.from_doc(tokenizer_path)

    text = '! hello I am linhengyang<|endoftext|>haha Iam back again..'
    encoded = tok.encode(text, allowed_special='all')
    text_ = tok.decode(encoded)
    print(encoded)
    assert text_ == text

    # 测试加载 模型(pure-torch)
    model_path = os.path.join(gpt2_resource_dir, 'gpt2.bin')
    net_state_dict = torch.load(model_path, map_location='cpu')
    print(net_state_dict.keys())

    config = gpt2Config()