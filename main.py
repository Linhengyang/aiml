import os
import warnings
warnings.filterwarnings("ignore")
from Code.projs.transformer._jobs import prepare_job, train_job, infer_job

if __name__ == "__main__":
    eng_symbols_path, fra_symbols_path = prepare_job()

    saved_params_fpath = train_job(eng_symbols_path, fra_symbols_path)

    infer_job(saved_params_fpath, eng_symbols_path)