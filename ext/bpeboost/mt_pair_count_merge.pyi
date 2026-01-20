import numpy as np
import typing as t

def initialize_thread(block_size: int) -> None:
    ...

def count_u16pair_batch(
    tokens_offsets,
    ):
    ...

def merge_u16pair_batch(
    tokens_offsets,
    pair_L:np.uint16,
    pair_R:np.uint16,
    new_token:np.uint16
    ):
    ...
    