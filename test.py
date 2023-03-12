import os
import warnings
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
from Code.projs.bert._jobs import pretrain_job, embed_job
if __name__ == "__main__":
    '''
    test code in this test.py file. After successful tests, code will be moved to _jobs.py under proj_name
    '''
    embed_job()