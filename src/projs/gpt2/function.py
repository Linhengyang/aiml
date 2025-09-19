# shared functions for GPT
import torch
from typing import Tuple
from ...core.base.functions.sequence import segments_to_positions


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