import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
import os
import regex as re
from ...core.utils.text.tokenizer import GPT4_TOKENIZER_REGEX, ENDOFTEXT

def encode_to_ints(s:str, encoding='utf-8') -> t.List[int]:
    return list( s.encode(encoding) )

def read_parquet_in_batches(file_path: str, batch_size: int, columns: t.List[str]):
    """
    从 Parquet 文件中分批读取数据。

    Args:
        file_path (str): Parquet 文件的路径。
        batch_size (int): 每次读取的行数。
        columns: 每次读取的列集（列名列表）

    Yields:
        pd.DataFrame: 包含当前批次数据的 pyarrow record batch。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' not found.")
    
    parquet_file = pq.ParquetFile(file_path)

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch


def write_tokens_parquet(output_path:str, data_batch_iter:t.Generator, text_column:str='text'):
    '''
    从data_batch_iter 中读取 batch of text, split 成 chunks 之后, 把每个 chunk encode 成 tokens(list of integers)
    把 tokens 写入到 Parquet 文件。per tokens-chunk per row

    tokens-chunk 是 ListArray 类型 --> list of int32 named 'token' --> list_of_ints_type
        变长的 List Array, field 是 token, dtype 为 int32（token的最大vocab size 十几万即可，int32值域足够）
    
    Parquet table 的 schema:
        列名: tokens
        column type: list_of_ints_type
    
    '''
    list_of_ints_type = pa.list_(pa.field('token', pa.int32()))
    tokens_pq_schema = pa.schema([
        pa.field('tokens', list_of_ints_type) # 列名 tokens, 列 data type: list_of_ints_type
    ])

    with pq.ParquetWriter(output_path, tokens_pq_schema, compression='snappy') as writer:
        for i, batch in enumerate(data_batch_iter):
            text_batch = ENDOFTEXT.join( batch[text_column].to_pylist() ) + ENDOFTEXT
            chunks_str = re.findall(GPT4_TOKENIZER_REGEX, text_batch) # list of tokens(string)
            chunks_tokens = [encode_to_ints(chunk) for chunk in chunks_str] # list of list of integers(every list of integers as tokens)

            # to build table, get offsets and values
            offset, current_offset = [0], 0
            for tokens in chunks_tokens:
                current_offset += len(tokens)
                offset.append(current_offset)

            values_array = pa.array([token for tokens in chunks_tokens for token in tokens], type=pa.int32())
            list_array = pa.ListArray.from_arrays(offset, values_array)

            # 创建 pa table
            batch_table = pa.Table.from_arrays([list_array], schema=tokens_pq_schema)
            writer.write_table(batch_table)