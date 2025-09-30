import os
import warnings
warnings.filterwarnings("ignore")
from src.projs.gpt2._jobs import env_set, pretrain_job, build_tokenizer_job, generate_job

if __name__ == "__main__":
    env_set()
    generate_job()


