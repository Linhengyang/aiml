# test.py
from src.core.design.producer_consumer import *
from src.core.utils.text.tokenizer import ByteTokenizer
cache_play = '../cache/playground/'
import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
import os
import regex as re
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
ENDOFTEXT = '<|endoftext|>'
GPT4_TOKENIZER_REGEX = \
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def encode_to_ints(s:str, encoding='utf-8') -> t.List[int]:
    return list( s.encode(encoding) )

def read_parquet_in_batches(file_path: str, batch_size: int):
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

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch['text']


if __name__ == "__main__":
    sample_pq = os.path.join(cache_play, "sample_raw.parquet")
    sample_tokens_pq = os.path.join(cache_play, "sample_tokens.parquet")
    batch_size = 3
    # 输出 tokens schema
    list_of_ints_type = pa.list_(pa.field('token', pa.int32()))
    output_schema = pa.schema([
        pa.field('tokens', list_of_ints_type) # 变长的整数列表
    ])
    # 把 raw parquet 文件 pre-split 成 chunks, 再把每个chunk encode 成 0-255 integers, 输出称为 parquet 文件.
    # 每一行是一个 tokens (list of integers)

    with pq.ParquetWriter(sample_tokens_pq, output_schema, compression='snappy') as writer:
        for i, text in enumerate(read_parquet_in_batches(sample_pq, batch_size)):
            string_batch = ENDOFTEXT.join(text.to_pylist())
            chunks = re.findall(GPT4_TOKENIZER_REGEX, string_batch) # list of tokens(string)

            chunks_tokens = [encode_to_ints(chunk) for chunk in chunks] # list of list of integers(every list of integers --> tokens)

            # 扁平化一个大列表，计算每个子列表的偏移量
            flat_tokens = [token for tokens in chunks_tokens for token in tokens]
            # offset
            offset = [0]
            current_offset = 0
            for tokens in chunks_tokens:
                current_offset += len(tokens)
                offset.append(current_offset)
            
            values_array = pa.array(flat_tokens, type=pa.int32())
            
            list_array = pa.ListArray.from_arrays(offset, values_array)

            # 创建 pa table
            batch_table = pa.Table.from_arrays([list_array], schema=output_schema)
            writer.write_table(batch_table)

    # 反向验证
    # sample_tokens = pd.read_parquet(sample_tokens_pq)
    # bt = []
    # for tokens in sample_tokens['tokens']:
    #     bt.extend(tokens)
    # bttok = ByteTokenizer()
    # print( bttok.decode(bt) )

    # sample_raw = pd.read_parquet(sample_pq)
    # text = []
    # for article in sample_raw['text']:
    #     text.append(article)
    # print(ENDOFTEXT.join(text))