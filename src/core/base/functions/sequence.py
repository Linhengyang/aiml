# sequence.py
import torch
import typing as t


def get_positions_from_segments(segments:torch.Tensor, origin_pos:int = 0, pad_seg:int|None = None, pad_pos:int = 0) -> torch.Tensor:
    '''
    origin_pos: smallest position for positions
    pad_seg: the segment ID in segments to represent PAD. if None, indicates NO PAD in segments
    pad_pos: the position ID in positions to represent PAD

    e.g,
    segments [0, 0, 1, 1, 1, 2, 3, 3] --> positions [0, 1, 0, 1, 2, 0, 0, 1]
    '''
    # 统一编序, PAD 位置也编
    # 将 PAD 位置的 position_id --set-to--> pad_pos
    # TODO



def get_segments_from_positions(positions:torch.Tensor, origin_pos:int = 0, pad_pos:int|None = None, pad_seg:int = 0) -> torch.Tensor:
    '''
    origin: smallest position for positions
    start: smallest segment id for segments

    e.g,
    positions [0, 1, 0, 1, 2 ,3, 0, 0, 1] --> segments [0, 0, 1, 1, 1, 1, 2, 3, 3]
    '''
    pass


