import os
import warnings
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
from Code.projs.recsys._jobs import mf_train_job, mf_infer_job, autorec_train_job, autorec_infer_job
from Code.projs.recsys.Dataset import d2lCTRDataset

if __name__ == "__main__":
    '''
    test code in this test.py file. After successful tests, code will be moved to _jobs.py under proj_name
    '''
    data_path = ['../../data/CTR/train.csv',]
    trainset = d2lCTRDataset(data_path)
    print(trainset.num_classes)
    for X, y in torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=True):
        print(X)
        print(y)
        break