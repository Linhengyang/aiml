# test.py
# import pyarrow as pa
# import pyarrow.parquet as pq
# import pyarrow.dataset as ds
# import regex as re
# import os
# import typing as t
import sys
from src.core.utils.text.tokenizer import mpbufferBBPE_u16Tokenizer


if __name__ == "__main__":
    # tokenizer = mpbufferBBPE_u16Tokenizer(name='test', buffer_dir="../cache/bpe_build/buffer/", explicit_n_vocab=256+3+5)
    # corpus = "aaabdaaabac"
    # tokenizer.train_bpe(3, corpora=corpus, column=None, format='text', language='en', batch_size_level='min', verbose=True)

    def test_merge_pair(tokens_flat, offsets, num_chunks, pair_L,pair_R, new_token):
        
        _LENGTH = offsets[num_chunks]
        merged_tokens_flat = [None]*_LENGTH
        merged_offsets = [None]*(num_chunks+1)
        merged_offsets[0] = 0

        num_merges = 0
        num_filtered = 0

        for k in range(1, num_chunks+1):
            j = offsets[k-1]
            while j < offsets[k]:
                if j < offsets[k]-1 and tokens_flat[j] == pair_L and tokens_flat[j+1] == pair_R:
                    merged_tokens_flat[j-num_merges-num_filtered] = new_token
                    j += 2
                    num_merges += 1
                else:
                    merged_tokens_flat[j-num_merges-num_filtered] = tokens_flat[j]
                    j += 1
            merged_offsets[k] = offsets[k] - num_merges-num_filtered

            if merged_offsets[k] - merged_offsets[k-1] == 1:
                merged_offsets[k] -= 1
                num_filtered += 1

        return merged_tokens_flat, merged_offsets
    
    corpus = "aaabdaaabac"
    # num_chunks = 3
    # tokens_flat = [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]
    # offsets = [0, 2, 3, 11]
    # pair_L, pair_R = 97, 97
    # new_token = 256

    num_chunks = 2
    tokens_flat = [97, 97, 97]
    offsets = [0, 1, 3]
    pair_L, pair_R = 97, 97
    new_token = 256

    print(test_merge_pair(tokens_flat, offsets, num_chunks, pair_L,pair_R, new_token))