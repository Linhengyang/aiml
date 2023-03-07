import os
import warnings
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
from Code.projs.word2vec._jobs import skipgram_train_job, cbow_train_job, skipgram_infer_job

if __name__ == "__main__":
    '''
    test code in this test.py file. After successful tests, code will be moved to _jobs.py under proj_name
    '''
    skipgram_infer_job()