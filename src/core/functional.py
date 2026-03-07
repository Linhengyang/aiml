import torch
import torch.nn as nn
from torch import Tensor
import typing as t

def onehot_concat_multifeatures(input_tensor: Tensor, num_classes: Tensor) -> Tensor:
    '''
    此函数将多个categorical features 分别onehot变换并在最后一维concat

    input:
    1. input_tensor:
    (*, num_features), with elements are level-index of categorical features
    The last dimention of input_tensor, is among different categorical features
    >>>
    tensor([[9, 3, 4],
            [0, 1, 0],
            [7, 0, 0],
            [0, 0, 0],
            [1, 1, 1]])
    where batch_size = 5, num_features = 3
    >>>
    tensor([9, 3, 4, 0])
    where num_features = 4
    >>>
    tensor([[[9, 3, 4],
             [0, 1, 0],
             [7, 0, 0],
             [0, 0, 0],
             [1, 1, 1]],
            [[9, 3, 4],
             [0, 1, 0],
             [7, 0, 0],
             [0, 0, 0],
             [1, 1, 1]]])
    where batch_size = 2, position_dims = 5, num_features = 3
    2. num_classes:
    (num_features, ), with elements are number of levels(classes) for every categorical feature
    
    len(num_classes) == input_tensor.shape[-1]

    return:
    onehot every catogorical feature along its own num_class, then concat all onehot vectors along on dim=-1
    shape: ( *, sum(num_classes) )
    '''
    assert len(num_classes) == input_tensor.shape[-1], 'every feature must have its num_class'
    assert torch.all(input_tensor < num_classes), \
        f'index number exceeds or be equal to num_classes. Index number must be smaller than corresponding num_class'
    offsets = torch.cat([torch.zeros(1,), torch.cumsum(num_classes, dim=0)[:-1]], dim=0).type(num_classes.dtype).to(input_tensor.device)
    return nn.functional.one_hot(input_tensor + offsets, num_classes.sum()).sum(dim=-2)


def segments_to_positions(segments:torch.Tensor, origin_pos:int = 0, pad_seg:int|None = 0, pad_pos:int = 0) -> torch.Tensor:
    '''
    origin_pos: starting position id for sequence. default 0
    pad_seg: the segment id in segments to represent PAD. None implies NO-PAD in segments. default 0
    pad_pos: the position id to represent PAD. default 0

    e.g,
    with other args as default: segments [0, 0, 1, 1, 1, 2, 3, 3] --> positions [0, 0, 0, 1, 2, 0, 0, 1]
    pad_seg=None, others as default: segments [0, 0, 1, 1, 1, 2, 3, 3] --> positions [0, 1, 0, 1, 2, 0, 0, 1]
    '''
    # 步骤一: 统一编序, PAD 位置也编
    B, L = segments.shape
    device = segments.device

    # default [0, 1, ..., L-1]
    offsets_origin = torch.arange(0, L, device=device).unsqueeze(0).expand(B, L) # [B, L]

    # 找到 序列中 每个 segment 的起始位置: [B, L]bool, True --> is start of a seq; False --> not start
    is_start = segments[:, 1:] != segments[:, :-1] # [B, L-1]bool
    is_start = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=device), is_start], dim=-1) # [B, L]bool

    # 找到 每个 segment 的起始位置 相对 origin_pos 的 offset
    seg_start_offsets = torch.where(is_start, offsets_origin, 0)
    segment_offsets = seg_start_offsets.cummax(dim=1).values
    positions = offsets_origin - segment_offsets + origin_pos # [B, L]

    # 步骤二：if pad_seg is not None, 将 PAD 位置的 position_id --set-to--> pad_pos
    if pad_seg is not None:
        positions = torch.where(segments != pad_seg, positions, pad_pos)

    return positions


def positions_to_segments(positions:torch.Tensor, origin_pos:int = 0, pad_pos:int|None = 0, pad_seg:int = 0) -> torch.Tensor:
    '''
    origin: smallest position for positions
    start: smallest segment id for segments

    e.g,
    positions [0, 1, 0, 1, 2 ,3, 0, 0, 1] --> segments [0, 0, 1, 1, 1, 1, 2, 3, 3]
    '''
    pass
