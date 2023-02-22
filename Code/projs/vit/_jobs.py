import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from .Dataset import FMNISTDatasetOnline
from .Network import ViTEncoder
from .Trainer import vitTrainer
from .Evaluator import vitEpochEvaluator
from .Predictor import fmnistClassifier
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
    num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens = 2, 8, 512, 0.1, 0.1, 1024
    img_shape, patch_size = (1, 96, 96), (16, 16)
    vit = ViTEncoder(img_shape, patch_size, num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens)
    loss = nn.CrossEntropyLoss(reduction='none')
    # init trainer
    trainer = vitTrainer(net=vit, loss=loss, num_epochs=10, batch_size=128)
    trainer.set_device(torch.device('cuda'))## set the device
    trainer.set_data_iter(trainset, validset, testset)## set the data iters
    trainer.set_optimizer(lr=0.1)## set the optimizer
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
    device = torch.device('cpu')
    # obtain pred batch
    path = '../../data'
    resize = (28, 28)
    validset = FMNISTDatasetOnline(path, False, resize)
    for img_batch, labels in validset:
        break
    # load model
    num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens = 2, 4, 512, 0.3, 0.4, 512
    img_shape, patch_size = (1, 28, 28), (7, 7)
    net = ViTEncoder(img_shape, patch_size, num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens)
    trained_net_path = os.path.join(local_model_save_dir, 'vit', 'vit_v1.params')
    net.load_state_dict(torch.load(trained_net_path, map_location=device))
    # init predictor
    classifier = fmnistClassifier(device)
    # predict
    print(classifier.predict(img_batch, labels, net))
    # evaluate
    print('accuracy: ', classifier.evaluate())
    print('pred scores: ', classifier.pred_scores)