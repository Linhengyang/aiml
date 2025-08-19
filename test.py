# test.py
import sys
import typing as t

cache_play = '../cache/playground/'


from src.core.utils.text.tokenizer import asyncBBPETokenizer
import pyarrow as pa

if __name__ == "__main__":
    token_dtype = pa.uint16()
    p_counts_schema = pa.schema([
        pa.field('L', token_dtype),
        pa.field('R', token_dtype),
        pa.field('counts', pa.uint64()),
        ])
    tokens_schema = pa.schema([
        pa.field( 'tokens', pa.large_list(pa.field('token', token_dtype)) ),
        ])
    
    def encode_to_ints(s:str, encoding='utf-8') -> t.List[int]:
        return list( s.encode(encoding) )
    
    chunks_tokens = [[1,2,3], [], [3,4,5], [1], [2,6,5,6], [1, 1]]

    table = pa.Table.from_pydict({tokens_schema[0].name: chunks_tokens}, tokens_schema)
    batch = table.to_batches()[0]

    tokens_flat = batch[tokens_schema[0].name].values.to_numpy()

    offsets = batch[tokens_schema[0].name].offsets.to_numpy()