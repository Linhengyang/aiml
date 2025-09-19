# shared functions for GPT
import torch
from typing import Tuple


def get_segs_pos_attnmask_train(
        input_segs: None|torch.Tensor, L_q: int,
        *,
        device) -> Tuple[torch.Tensor|None, torch.Tensor, torch.Tensor|None]:
    
    segments = input_segs

    if segments is None:
        positions = torch.arange(0, L_q, dtype=torch.long, device=device).unsqueeze(0) # [1, L_q]
        attention_mask = None
    else:
        # train mode
        # positions: PAD pos --> 0, TOKEN pos --> index from 0 in relevent sequence
        # TODO
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
        # positions: PAD pos --> 0, TOKEN pos --> index from 0 in global sequence
        # TODO
        # attention_mask: qk any PAD --> False, qk no PAD --> True
        is_pad = segments != 0 # [B, L_q]bool, PAD -> false, nonPAD -> true
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

    segments = None if past_segs == input_segs == None  else torch.cat([past_segs, input_segs], dim=-1)
    
    if segments is None:
        positions = torch.arange(0, L_past+L_q, dtype=torch.long, device=device).unsqueeze(0) # [1, L_so_far]
        attention_mask = None
    else:
        # infer.decode mode
        # positions: PAD pos --> 0, TOKEN pos --> index from 0 in global sequence
        # TODO
        # attention_mask: qk any PAD --> False, qk no PAD --> True
        is_q_pad = input_segs != 0 # [B, L_q]bool, PAD -> false, nonPAD -> true
        is_pad = segments != 0 # [B, L_so_far]bool, PAD -> false, nonPAD -> true
        attention_mask = is_q_pad.unsqueeze(-1) * is_pad.unsqueeze(-2) # [B, L_q, L_so_far]]bool, qk any PAD -> false, qk no PAD -> true
    
    #  [B, L_so_far] [B, L_so_far]   [B, L_q, L_so_far]
    return segments, positions, attention_mask