import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from .Dataset import FMNISTDatasetOnline
from .Network import ViTEncoder
from .Trainer import vitTrainer
from .Evaluator import vitEpochEvaluator
import yaml
configs = yaml.load(open('Code/projs/vit/configs.yaml', 'rb'), Loader=yaml.FullLoader)
local_model_save_dir = configs['local_model_save_dir']


def train_job():
    # build datasets
    path = '../../data'
    resize = (28, 28)
    trainset = FMNISTDatasetOnline(path, True, resize)
    validset = FMNISTDatasetOnline(path, False, resize)
    testset = FMNISTDatasetOnline(path, False, resize)
    # design net & loss
    num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens = 2, 4, 512, 0.3, 0.4, 512
    img_shape, patch_size = (1, 28, 28), (7, 7)
    vit = ViTEncoder(img_shape, patch_size, num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens)
    loss = nn.CrossEntropyLoss(reduction='none')
    # init trainer
    trainer = vitTrainer(net=vit, loss=loss, num_epochs=10, batch_size=128)
    trainer.set_device(torch.device('cuda'))## set the device
    trainer.set_data_iter(trainset, validset, testset)## set the data iters
    trainer.set_optimizer(lr=0.0005)## set the optimizer
    trainer.set_epoch_eval(vitEpochEvaluator('train_logs.txt', visualizer=True))## set the epoch evaluator
    # start
    trainer.log_topology('lazy_topo.txt')## print the vit topology
    check_flag = trainer.resolve_net(need_resolve=True)## check the net & loss
    if check_flag:
        trainer.log_topology('def_topo.txt')## print the defined topology
        trainer.init_params()## init params
    # fit
    trainer.fit()
    # save
    trainer.save_model('vit_test.params')

def infer_job():
    pass