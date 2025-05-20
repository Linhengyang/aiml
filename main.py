import os
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # transformer

    from Code.projs.transformer._jobs import prepare_job, train_job, infer_job
    eng_vocab_path, fra_vocab_path = prepare_job()

    # saved_params_fpath = train_job(eng_symbols_path, fra_symbols_path)
    # saved_params_fpath = "../model/text_translator/saved_params_2025-05-08_17:32.pth"

    # infer_job(saved_params_fpath, eng_symbols_path)

    # transformer tested

    ###########################################################

    # # vit

    # from Code.projs.vit._jobs import prepare_job, train_job, infer_job
    
    # # prepare_job()

    # # saved_params_fpath = train_job("local")
    # saved_params_fpath = "../model/vit/saved_params_2025-05-09_16:36.pth"

    # infer_job(saved_params_fpath)
    

    # # vit tested