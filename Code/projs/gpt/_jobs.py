import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import yaml


configs = yaml.load(open('Code/projs/gpt/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################
# set train log file path / network resolve output path / params save path / source&targe vocabs path

################## params saved in workspace/model ##################
model_proj_dir = os.path.join( configs['model_dir'], configs['proj_name'] )



################## log file in workspace/logs ##################
log_proj_dir = os.path.join( configs['log_dir'], configs['proj_name'] )



################## data-params ##################



################## network-params ##################



################## train-params ##################





def prepare_job():
    print('prepare job begin')
    pass
    outputs = 1
    print('prepare job complete')
    return outputs




def train_job(outputs, *args, **kwargs):
    print('train job begin')
    pass
    saved_params_fpath = ''
    print('train job complete')
    return saved_params_fpath




def infer_job(saved_params_fpath, *args, **kwargs):
    print('infer job begin')
    pass
    print('infer job complete')