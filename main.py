import os
import warnings
warnings.filterwarnings("ignore")
from src.projs.transformer._jobs import prepare_job, train_job, infer_job

if __name__ == "__main__":
    eng_vocab_path, fra_vocab_path = prepare_job()
    saved_params_fpath = train_job(eng_vocab_path, fra_vocab_path)
    infer_job(saved_params_fpath, eng_vocab_path, fra_vocab_path)