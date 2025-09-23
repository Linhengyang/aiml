import os
import warnings
warnings.filterwarnings("ignore")
from src.projs.gpt2._jobs import test_job, build_tokenizer_job

if __name__ == "__main__":
    build_tokenizer_job()



