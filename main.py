import os
import warnings
warnings.filterwarnings("ignore")
from src.apps.bpe_build._jobs import bpe_prepare, bpe_build, bpe_continue

if __name__ == "__main__":
    bpe_prepare()
    # tok_fpath, vocab_cache_dir = bpe_build()