# shared functions for GPT
import torch
from typing import Tuple
from ...core.utils.data.transform import segments_to_positions

# train mode 下, 从 input 的 segments(可以为None) 按以下标准生成对应的 positions 和 attention mask
# 若 input_segs = None, 默认 input 属于同一个text且没有PAD, 则 segments = None, positions = 从0开始总长度为L_q的位置编码 [1, L_q], attention_mask = None
# 若 input_segs != None, 那么默认其中 0 代表PAD, 位置编码赋0; 从 0 开始给每个单一 seg代表的单一序列 独立 赋位置编码, tensor [B, L_q]long
# attention_mask 是 tensor [B, L_q, L_q]long, 只有 qk 都非PAD 且 qk 的 segment 相等 才为 True, 其余都是 False
def get_segs_pos_attnmask_train(
        input_segs: None|torch.Tensor, L_q: int,
        *,
        device) -> Tuple[torch.Tensor|None, torch.Tensor, torch.Tensor|None]:
    '''
    实际的 input_segs 是 batch. e.g 只是取一行作例

    input:
        input_segs: None | tensor[B, L]long, where 0 --> PAD, 1/2/... --> TEXT ID
        e.g, [0, 0, 1, 1, 1, 2, 2]

    return:
        segments: same as input_segs
        e.g, [0, 0, 1, 1, 1, 2, 2]

        positions: tensor[B, L]long
            if input_segs is None: --> 0 to L-1 as positions
            else: position id where PAD pos --> 0, TOKEN pos --> index(starting from 0) inside every sequence
        e.g, [0, 0, 0, 1, 2, 0, 1]

        attention_mask: None | tensor[B, L, L]bool
            if input_segs is None: --> None
            else: tensor[B, L, L]bool, qk any PAD or irrelevent --> False, qk no PAD and relevent --> True
        e.g,   0  0  1  1  1  2  2 
            0  F  F  F  F  F  F  F
            0  F  F  F  F  F  F  F
            1  F  F  T  T  T  F  F
            1  F  F  T  T  T  F  F
            1  F  F  T  T  T  F  F
            2  F  F  F  F  F  T  T
            2  F  F  F  F  F  T  T
    '''
    segments = input_segs

    if segments is None:
        positions = torch.arange(0, L_q, dtype=torch.long, device=device).unsqueeze(0) # [1, L_q]
        attention_mask = None
    else: # segments [B, L]
        # train mode
        # positions: PAD pos --> 0, TOKEN pos --> index from 0 in relevent sequence
        positions = segments_to_positions(segments) # same device with segments
        # attention_mask: qk any PAD or irrelevent --> False, qk no PAD and relevent --> True
        is_pad = segments != 0 # [B, L_q]bool, PAD -> false, nonPAD -> true
        pad_mask = is_pad.unsqueeze(-1) * is_pad.unsqueeze(-2) # [B, L_q, L_q]bool, qk any PAD -> false, qk no PAD -> true
        relevent_mask = segments.unsqueeze(-1) == segments.unsqueeze(-2) # [B, L_q, L_q]bool, qk irrelevent -> false, qk relevent -> true
        attention_mask = pad_mask * relevent_mask # [B, L_q, L_q]
    
    #     [B, L_q]   [B, L_q]   [B, L_q, L_q]
    return segments, positions, attention_mask




# infer.prefill mode 下, 从 input 的 segments(可以为None) 按以下标准生成对应的 positions 和 attention mask
# 若 input_segs = None, 默认 input 属于同一个text且没有PAD, 则 segments = None, positions = 从0开始总长度为L_q的位置编码 [1, L_q], attention_mask = None
# 若 input_segs != None, 那么默认其中 0 代表PAD, 位置编码赋0; 从 0 开始给所有非 PAD 位置按序 赋位置编码, tensor [B, L_q]long
# attention_mask 是 tensor [B, L_q, L_q]long, 只有 qk 都非PAD 才为 True, 其余都是 False
def get_segs_pos_attnmask_prefill(
        input_segs: None|torch.Tensor, L_q: int,
        *,
        device) -> Tuple[torch.Tensor|None, torch.Tensor, torch.Tensor|None]:
    
    segments = input_segs

    if segments is None:
        positions = torch.arange(0, L_q, dtype=torch.long, device=device).unsqueeze(0) # [1, L_q]
        attention_mask = None
    else:
        # infer.prefill mode
        is_pad = segments != 0 # [B, L_q]bool, PAD -> false, nonPAD -> true
        # positions: PAD pos --> 0, TOKEN pos --> index from 0 in global sequence
        positions = segments_to_positions(is_pad.to(torch.long)) # same device with segments
        # attention_mask: qk any PAD --> False, qk no PAD --> True
        attention_mask = is_pad.unsqueeze(-1) * is_pad.unsqueeze(-2) # [B, L_q, L_q]bool, qk any PAD -> false, qk no PAD -> true
    
    #     [B, L_q]   [B, L_q]   [B, L_q, L_q]
    return segments, positions, attention_mask




# infer.decode mode 下, 从 input 的 input_segs(可以为None) 和 kv_cache 的 past_segs(可以为None) 按以下标准生成对应的 positions 和 attention mask
# step1: 确定 segments: 包含 past + input 的 segments
#        若 二者都为 None, 则 segments = None; 若二者有其一不为None, 则视另外一个为全1(即非PAD单一序列); 若二者皆存在, 则OK

# 若 segments = None, 则 segments = None, positions = 从0开始总长度为 L_q+L_past 的位置编码 [1, L_q+L_past], attention_mask = None
# 若 segments != None, 那么默认其中 0 代表PAD, 位置编码赋0; 从 0 开始给所有非 PAD 位置按序 赋位置编码, tensor [B, L_q+L_past]long
# attention_mask 是 tensor [B, L_q, L_q+L_past]long, 只有 qk 都非PAD 才为 True, 其余都是 False
def get_segs_pos_attnmask_decode(
        input_segs: None|torch.Tensor, L_q: int, past_segs: None|torch.Tensor, L_past: int,
        *,
        device) -> Tuple[torch.Tensor|None, torch.Tensor, torch.Tensor|None]:

    if past_segs is None and input_segs is None:
        pass
    elif past_segs is not None and input_segs is None:
        # input_segs 视作全 1. 即 nonPAD
        input_segs =  torch.ones(past_segs.size(0), L_q, dtype=torch.long, device=device)
    elif past_segs is None and input_segs is not None:
        # past_segs 视作全 1. 即 nonPAD
        past_segs = torch.ones(input_segs.size(0), L_past, dtype=torch.long, device=device)
    else:
        pass

    segments = None if past_segs is input_segs is None  else torch.cat([past_segs, input_segs], dim=-1)
    
    if segments is None:
        positions = torch.arange(0, L_past+L_q, dtype=torch.long, device=device).unsqueeze(0) # [1, L_so_far]
        attention_mask = None
    else:
        # infer.decode mode
        is_pad = segments != 0 # [B, L_so_far]bool, PAD -> false, nonPAD -> true
        # positions: PAD pos --> 0, TOKEN pos --> index from 0 in global sequence
        positions = segments_to_positions(is_pad.to(torch.long)) # same device with segments
        # attention_mask: qk any PAD --> False, qk no PAD --> True
        is_q_pad = input_segs != 0 # [B, L_q]bool, PAD -> false, nonPAD -> true
        attention_mask = is_q_pad.unsqueeze(-1) * is_pad.unsqueeze(-2) # [B, L_q, L_so_far]]bool, qk any PAD -> false, qk no PAD -> true
    
    #  [B, L_so_far] [B, L_so_far]   [B, L_q, L_so_far]
    return segments, positions, attention_mask