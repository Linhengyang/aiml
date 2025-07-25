import numpy as np
import typing as t
def allocate_memory(block_size: int) -> None:...
def merge_pair_batch(
    tokens_offsets:tuple[np.ndarray, np.ndarray],
    pair_L:np.uint16,
    pair_R:np.uint16,
    new_token:np.uint16
    ) -> tuple[np.ndarray, np.ndarray]:
    ...
def release_memory():...
