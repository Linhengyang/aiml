import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import yaml
from .Dataset import skipgramDataset, cbowDataset
from .Network import skipgramNegSp, cbowNegSp
from .Trainer import word2vecTrainer
from .Evaluator import word2vecEpochEvaluator
configs = yaml.load(open('Code/projs/word2vec/configs.yaml', 'rb'), Loader=yaml.FullLoader)
local_model_save_dir = configs['local_model_save_dir']
base_data_dir = configs['base_data_dir']
word2vec_dir = configs['word2vec_dir']
ptb_train_fname = configs['ptb_train_fname']

def skipgram_train_job():
    trainset = skipgramDataset(os.path.join(base_data_dir, word2vec_dir, ptb_train_fname))
    vocab = trainset.vocab
    # design net & loss
    embed_size = 100
    net = skipgramNegSp(len(vocab), embed_size)
    class WeightedSigmoidBCELoss(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, input, target, weight):
            # input shape: (batch_size, *)
            # target shape: (batch_size, *)
            out = nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction='none')
            return out.mean(dim=1) # (batch_size, )
    loss = WeightedSigmoidBCELoss()
    # init trainer for num_epochs & batch_size & learning rate
    num_epochs, batch_size, lr = 100, 128, 0.00015
    trainer = word2vecTrainer(net, loss, num_epochs, batch_size)
    trainer.set_device(torch.device('cuda'))## set the device
    trainer.set_data_iter(trainset)## set the data iters
    trainer.set_optimizer(lr)## set the optimizer
    trainer.set_epoch_eval(word2vecEpochEvaluator(num_epochs, log_fname='train_logs.txt', visualizer=False))
    # start
    trainer.log_topology('skipgram_topo.txt')## print the vit topology
    check_flag = trainer.resolve_net(need_resolve=False)## check the net & loss
    # fit
    trainer.fit()
    # save
    trainer.save_model('skipgram_v1.params')

def cbow_train_job():
    trainset = cbowDataset(os.path.join(base_data_dir, word2vec_dir, ptb_train_fname))
    vocab = trainset.vocab
    # design net & loss
    embed_size = 100
    net = cbowNegSp(len(vocab), embed_size)
    class SigmoidBCELoss(nn.Module):
        def __init__(self) -> None:
            super().__init__()
        def forward(self, input, target, *args, **kwargs):
            # input shape: (batch_size, *)
            # target shape: (batch_size, *)
            out = nn.functional.binary_cross_entropy_with_logits(input, target, reduction='none')
            return out.mean(dim=1) # (batch_size, )
    loss = SigmoidBCELoss()
    # init trainer for num_epochs & batch_size & learning rate
    num_epochs, batch_size, lr = 100, 128, 0.00015
    trainer = word2vecTrainer(net, loss, num_epochs, batch_size)
    trainer.set_device(torch.device('cuda'))## set the device
    trainer.set_data_iter(trainset)## set the data iters
    trainer.set_optimizer(lr)## set the optimizer
    trainer.set_epoch_eval(word2vecEpochEvaluator(num_epochs, log_fname='train_logs.txt', visualizer=False))
    # start
    trainer.log_topology('cbow_topo.txt')## print the vit topology
    check_flag = trainer.resolve_net(need_resolve=False)## check the net & loss
    # fit
    trainer.fit()
    # save
    trainer.save_model('cbow_v1.params')