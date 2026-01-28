import os
import warnings
warnings.filterwarnings("ignore")
from src.projs.transformer._jobs import prepare, pretrain, translate
# from src.apps.bpe_build._jobs import bpe_continue, bpe_train
import time

if __name__ == "__main__":
    # env_set()
    # generate_job()
    start = time.time()
    saved_params_fpath = '../model/transformer/saved_params_2026-01-28_14:32.pth'
    vocab_path = '../artifact/transformer/vocab.json'
    translate(saved_params_fpath, vocab_path)
    
    end = time.time()
    print(f'time usage {end-start}')