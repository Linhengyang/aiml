import os
import warnings
warnings.filterwarnings("ignore")
from src.apps.bpe_build._jobs import bpe_prepare, bpe_train, bpe_continue

if __name__ == "__main__":
    bpe_prepare()
    # tok_path = '../artifact/bpe_build/tokenizer/tinyTok_7.tok'
    bpe_train(32000, 'tinyTok')
    # tok_fpath, vocab_cache_dir = bpe_build()