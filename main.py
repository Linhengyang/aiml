import os
import warnings
warnings.filterwarnings("ignore")
from Code.projs.transformer._jobs import prepare_job, train_job, infer_job

if __name__ == "__main__":
    # eng_symbols_path, fra_symbols_path = prepare_job()

    eng_symbols_path, fra_symbols_path = "../cache/text_translator/symbols/source.json", "../cache/text_translator/symbols/target.json"

    # saved_params_fpath = train_job(eng_symbols_path, fra_symbols_path)
    saved_params_fpath = "../model/text_translator/saved_params_2025-04-14_14:54.pth"

    infer_job(saved_params_fpath, eng_symbols_path)