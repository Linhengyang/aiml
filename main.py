import os
import warnings
warnings.filterwarnings("ignore")
# from src.projs.gpt2._jobs import env_set, pretrain_job, build_tokenizer_job, generate_job
from src.apps.bpe_build._jobs import bpe_continue
import time

if __name__ == "__main__":
    # env_set()
    # generate_job()
    start = time.time()
    
    bpe_continue(None)

    end = time.time()
    print(f'time usage {end-start}')