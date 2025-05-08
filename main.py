import os
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # # transformer

    # from Code.projs.transformer._jobs import prepare_job, train_job, infer_job
    # eng_symbols_path, fra_symbols_path = prepare_job()
    # # eng_symbols_path, fra_symbols_path = "../cache/text_translator/symbols/source.json", "../cache/text_translator/symbols/target.json"

    # # saved_params_fpath = train_job(eng_symbols_path, fra_symbols_path)
    # saved_params_fpath = "../model/text_translator/saved_params_2025-05-08_17:32.pth"

    # infer_job(saved_params_fpath, eng_symbols_path)

    # # transformer tested

    # vit

    from Code.projs.vit._jobs import prepare_job, train_job, infer_job
    
    prepare_job()

    saved_params_fpath = train_job("local")

    infer_job(saved_params_fpath)
    

    # transformer tested