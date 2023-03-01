import os
import warnings
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
from Code.projs.recsys._jobs import mf_train_job, mf_infer_job, autorec_train_job, autorec_infer_job
from Code.Utils.Common.DataTransform import CategDataParser
if __name__ == "__main__":
    '''
    test code in this test.py file. After successful tests, code will be moved to _jobs.py under proj_name
    '''
    data_path = ['../../data/CTR/train.csv', '../../data/CTR/test.csv']
    ctr_data_parser = CategDataParser(data_path, [1, 34])
    data = ctr_data_parser.mapped_offset_data
    print(data[0])
    print(data.shape)