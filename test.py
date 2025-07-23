# test.py
import sys

cache_play = '../cache/playground/'


from src.core.utils.text.tokenizer import asyncBBPETokenizer


if __name__ == "__main__":
    from merge_pair import merge_pair_batch
    tok = asyncBBPETokenizer(name='test', buffer_dir='.')
    tok._set_config(50, merge_pair_batch)
    import pickle
    pickle.dumps(
        (tok._write_batch, tok._func_merge_pair_batch)
        )