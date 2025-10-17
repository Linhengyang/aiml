huggingface gpt2 的 tokenizer 经过 byte-to-token 映射处理, 以准确展示 不可打印字符（以及空格）到可渲染字符


# hf tokenizer
tokenizer_path = os.path.join(gpt2_resource_dir, 'tokenizer.json')
with open(tokenizer_path) as f:
    hf_tok = json.load(f)

hf_tok['version'] # 1.0

hf_tok['truncation'] # None

hf_tok['padding'] # None

hf_tok['added_tokens']
[{'id': 50256, 'special': True, 'content': '<|endoftext|>', 'single_word': False, 'lstrip': False, 'rstrip': False, 'normalized': True}]

hf_tok['normalizer'] # None

hf_tok['pre_tokenizer'] # {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': True}

hf_tok['post_processor'] # {'type': 'ByteLevel', 'add_prefix_space': True, 'trim_offsets': False}

hf_tok['decoder'] # {'type': 'ByteLevel', 'add_prefix_space': True, 'trim_offsets': True}

hf_tok['model']
['dropout', 'unk_token', 'continuing_subword_prefix', 'end_of_word_suffix', 'fuse_unk', 'vocab', 'merges']

hf_tok['model']['dropout'] # None

hf_tok['model']['unk_token'] # None

hf_tok['model']['continuing_subword_prefix'] # 空字符串

hf_tok['model']['end_of_word_suffix'] # 空字符串

hf_tok['model']['fuse_unk'] # False

// byte-to-unicode转换: byte 0-255 中, 有 188 个可打印byte 直接映射到 自身字符, 另外 68 个 控制字符及空格, 映射到 68 个可打印unicode 字符(256-323)
// 这样保证任何一个 bytes 序列经过 b2u 映射后, 得到的是可打印的 unicode 字符序列
// 同时由于这个映射是双射, 任何 可打印的unicode字符序列经过 u2b 映射后, 得到的是 bytes 序列 

// vocab: bytes sequence --> ID 的映射表. 这里 bytes sequence 作了byte-to-unicode转换
// ID 0-255 是 188个 可打印字符+68个unicode字符(经过了重排), ID >=256 开始 是 byte-to-unicode mapped unicode chars --> ID
for k, v in hf_tok['model']['vocab'].items(): # 50257 个 token


// merges: Left bytes sequence, Right bytes sequence. 这里 bytes sequence 作了byte-to-unicode转换
// Left + Right --> merged bytes sequence. merged bytes sequence 是 vocab 从 256 开始的 bytes sequence
for i, v in enumerate(hf_tok['model']['merges']): # 50000 个 pair merges




# state dict

huggingface:
['wte.weight',
 'wpe.weight',

 o --> 11
    'h.0.ln_1.weight', 'h.0.ln_1.bias',
    'h.0.attn.bias',                                        ---> 这个是 casual 下三角, 是陈年旧设计. 现代用 动态生成 or 缓存但不保存的 casual_mask 替代, 甚至把因果融合进算子里
    'h.0.attn.c_attn.weight', 'h.0.attn.c_attn.bias',       ---> c_代表 conv1d, 是陈年旧设计. 与 linear(out_dim, in_dim) 是转置的关系
    'h.0.attn.c_proj.weight', 'h.0.attn.c_proj.bias',       ---> c_代表 conv1d, 是陈年旧设计
    'h.0.ln_2.weight', 'h.0.ln_2.bias',
    'h.0.mlp.c_fc.weight', 'h.0.mlp.c_fc.bias',             ---> c_代表 conv1d, 是陈年旧设计
    'h.0.mlp.c_proj.weight', 'h.0.mlp.c_proj.bias','        ---> c_代表 conv1d, 是陈年旧设计

 'ln_f.weight', 'ln_f.bias'
                                                            ---> 缺一个 wte 的 reverse-embedding 层
]


custom:
['W_tok_embd.weight',
 'W_pos_embd.weight',

 0 --> 11
    'blocks.0.layer_norm1.weight', 'blocks.0.layer_norm1.bias',

    'blocks.0.casual_attention.W_qkv.weight', 'blocks.0.casual_attention.W_qkv.bias',
    'blocks.0.casual_attention.W_o.weight', 'blocks.0.casual_attention.W_o.bias',
    'blocks.0.layer_norm2.weight', 'blocks.0.layer_norm2.bias',
    'blocks.0.gelu_ffn.W_in.weight', 'blocks.0.gelu_ffn.W_in.bias',
    'blocks.0.gelu_ffn.W_out.weight', 'blocks.0.gelu_ffn.W_out.bias',

 'layer_norm_final.weight', 'layer_norm_final.bias',
 'head_tok.weight'
]