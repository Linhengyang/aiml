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
    
    pretrain()

    end = time.time()
    print(f'time usage {end-start}')