# state_dict_adapt 里修改 huggingface/transformers 库标准的 model state_dict(weights), 以适应并载入 custom network
from collections import OrderedDict
from torch import Tensor
import re


def gpt2_state_dict_adaptor(hf_state_dict: OrderedDict[str, Tensor]) -> OrderedDict[str, Tensor]:
    my_state_dict = OrderedDict()

    for k, v in hf_state_dict.items():
        names = k.split('.')
        if names[0] == 'h' and names[2] == 'attn' and names[3] == 'bias':
            continue

        new_k = k.replace('wte.', 'W_tok_embd.').replace('wpe.', 'W_pos_embd.')
        new_k = new_k.replace('h.', 'blocks.').replace('ln_1.', 'layer_norm1.').replace('ln_2.', 'layer_norm2.')
        new_k = new_k.replace('attn.c_attn.', 'casual_attention.W_qkv.').replace('attn.c_proj.', 'casual_attention.W_o.')
        new_k = new_k.replace('mlp.c_fc.', 'gelu_ffn.W_in.').replace('mlp.c_proj.', 'gelu_ffn.W_out.')
        new_k = new_k.replace('ln_f.', 'layer_norm_final.')

        # c_ 开头的陈年旧设计是使用 conv1D 来实现 Linear 映射的, conv1D 权重(in_dim, out_dim)要转置成 linear 权重(out_dim, in_dim)
        if names[-2] in ('c_attn', 'c_proj', 'c_fc') and v.ndim == 2:
            v = v.t().contiguous()
        
        my_state_dict[new_k] = v

    my_state_dict['head_tok.weight'] = hf_state_dict['wte.weight']
    return my_state_dict