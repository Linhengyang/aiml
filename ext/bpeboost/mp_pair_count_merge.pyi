import numpy as np
import typing as t

def initialize_process(block_size: int) -> None:
    ...

def count_u16pair_batch(
    tokens_offsets_border,
    ):
    ...

def merge_u16pair_batch(
    tokens_offsets_border,
    pair_L:np.uint16,
    pair_R:np.uint16,
    new_token:np.uint16
    ):
    ...
def merge_u16pair_batch_v2(
    tokens_offsets_border,
    pair_L:np.uint16,
    pair_R:np.uint16,
    new_token:np.uint16
    ):
    ...
def close_process() -> None:
    ...
    