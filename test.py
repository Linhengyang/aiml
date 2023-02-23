import os
import warnings
import torch
warnings.filterwarnings("ignore")
from Code.projs.transformer._jobs import train_job, infer_job

if __name__ == "__main__":
    '''
    test code in this test.py file. After successful tests, code will be moved to _jobs.py under proj_name
    '''
    infer_job()