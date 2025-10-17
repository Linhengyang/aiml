# test.py
import torch
import typing as t
import os
import math
import json

temp = '../cache/temp/'
gpt2_resource_dir = '../../resource/llm/gpt/gpt2/'



from src.kits.huggingface.tokenizer_adapt import gpt2Tokenizer
from src.projs.gpt2.network import gpt2, gpt2Config


if __name__ == "__main__":
    ## 测试 tokenizer
    # tokenizer_path = os.path.join(gpt2_resource_dir, 'tokenizer.json')

    # tok = gpt2Tokenizer()
    # tok.from_doc(tokenizer_path)

    # text = '! hello I am linhengyang<|endoftext|>haha Iam back again..'
    # encoded = tok.encode(text, allowed_special='all')
    # text_ = tok.decode(encoded)
    # print(encoded)
    # assert text_ == text

    # 测试加载 模型(pure-torch)
    model_path = os.path.join(gpt2_resource_dir, 'gpt2.bin')
    net_state_dict = torch.load(model_path, map_location='cpu')
    print(type(net_state_dict))
    # 删除 

    config = gpt2Config(
        embd_size = 768,
        vocab_size = 50257, 
        embd_p_drop = 0.1,
        num_head = 12,
        use_bias = True,
        max_context_size = 1024,
        attn_p_drop = 0.1,
        resid_p_drop = 0.1,
        use_cached_casual_mask = True,
        use_rope = False,
        num_block = 1
        )
    net_init = gpt2(config)
    # print(net_init.state_dict().keys())
    print(type(net_init.state_dict()))