from .init_tokens import read_parquet_in_batches, write_tokens_parquet, GPT4_TOKENIZER_REGEX, ENDOFTEXT, encode_to_ints
import yaml
import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
import os
import regex as re





configs = yaml.load(open('src/apps/bpe_build/configs.yaml', 'rb'), Loader=yaml.FullLoader)
train_pq = configs['train_pq']
valid_pq = configs['valid_pq']

def corpus_to_init_tokens_pq(corpus, output_path):
    if not output_path.endswith('.parquet'):
        raise ValueError(f'output file must end with .parquet')

    list_of_ints_type = pa.list_(pa.field('token', pa.int32()))
    tokens_pq_schema = pa.schema([
        pa.field('tokens', list_of_ints_type) # 列名 tokens, 列 data type: list_of_ints_type
    ])

    with pq.ParquetWriter(output_path, tokens_pq_schema, compression='snappy') as writer:
        text_batch = corpus + ENDOFTEXT
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