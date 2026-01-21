import numpy as np
import typing as t

def initialize_process(block_size: int) -> None:
    ...

def count_u16pair_batch(
    tokens_offsets: tuple,
    ) -> tuple:
    ...

def merge_u16pair_batch(
    tokens_offsets: tuple,
    pair_L:np.uint16,
    pair_R:np.uint16,
    new_token:np.uint16
    ) -> tuple:
    ...

def close_process() -> None:
    ...
    