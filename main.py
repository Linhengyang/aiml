import os
import warnings
warnings.filterwarnings("ignore")
from src.projs.vit._jobs import prepare_job, train_job, infer_job

if __name__ == "__main__":
    prepare_job()
    saved_params_fpath = train_job('local')
    infer_job(saved_params_fpath)