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

cache_play = '../cache/playground/'





if __name__ == "__main__":
    # sample_pq = os.path.join(cache_play, "sample_raw.parquet")
    # sample_tokens_pq = os.path.join(cache_play, "sample_tokens.parquet")
    # batch_size = 3

    # # 测试 parquet write 是否覆盖原文件
    # pd.read_parquet(sample_tokens_pq)
    p_counts = {(1,2):10, (3,4):20}
    import json
    json.dump()