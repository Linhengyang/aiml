import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import yaml
configs = yaml.load(open('Code/projs/bert/configs.yaml', 'rb'), Loader=yaml.FullLoader)
local_model_save_dir = configs['local_model_save_dir']
base_data_dir = configs['base_data_dir']

def train_job():
    pass

def infer_job():
    pass