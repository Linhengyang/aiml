import numpy as np
import typing as t

def initialize(block_size: int) -> None:
    ...

def count_pair_batch(
    tokens_offsets_border,
    ):
    ...

def merge_pair_batch(
    tokens_offsets_border,
    pair_L:np.uint16,
    pair_R:np.uint16,
    new_token:np.uint16
    ):
    ...

def close() -> None:
    ...
