import os
import warnings
warnings.filterwarnings("ignore")
from src.apps.bpe_build._jobs import bpe_prepare, bpe_train, bpe_continue
from src.core.utils.common.performance import timeit

if __name__ == "__main__":
    bpe_prepare()
    with timeit():
        # tok_path=None表示不load, 从0开始续train
        bpe_continue(tok_path=None)



