# test.py
from src.core.design.producer_consumer import *
import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
import os
import regex as re
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
import pandas as pd


ENDOFTEXT = '<|endoftext|>'

GPT4_TOKENIZER_REGEX = \
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

cache_play = '../cache/playground/'

def encode_to_ints(s:str, encoding='utf-8') -> t.List[int]:
    return list( s.encode(encoding) )





def read_parquet_in_batches(file_path: str, batch_size: int, columns: list):
    """
    从 Parquet 文件中分批读取数据。

    Args:
        file_path (str): Parquet 文件的路径。
        batch_size (int): 每次读取的行数。

    Yields:
        pd.DataFrame: 包含当前批次数据的 Pandas DataFrame。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' not found.")
    
    parquet_file = pq.ParquetFile(file_path)

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch





if __name__ == "__main__":
    sample_pq = os.path.join(cache_play, "sample_raw.parquet")
    sample_tokens_pq = os.path.join(cache_play, "sample_tokens.parquet")
    batch_size = 3

    # 测试 parquet write 是否覆盖原文件
    pd.read_parquet(sample_tokens_pq)