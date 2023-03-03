import os
import warnings
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
from Code.projs.recsys._jobs import mf_train_job, mf_infer_job, autorec_train_job, autorec_infer_job, fm_train_job, deepfm_train_job

if __name__ == "__main__":
    '''
    test code in this test.py file. After successful tests, code will be moved to _jobs.py under proj_name
    '''
    deepfm_train_job()