import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from .Dataset import wikitextDataset
from .Network import BERT
from .Trainer import bertTrainer
from .Evaluator import bertEpochEvaluator
import yaml
configs = yaml.load(open('Code/projs/bert/configs.yaml', 'rb'), Loader=yaml.FullLoader)
local_model_save_dir = configs['local_model_save_dir']
base_data_dir = configs['base_data_dir']

def train_job():
    train_path, max_len = os.path.join('../../data', 'bert/wikitext-2', 'wiki.train.tokens'), 64
    test_path = os.path.join('../../data', 'bert/wikitext-2', 'wiki.test.tokens')
    trainset = wikitextDataset(train_path, max_len)
    testset = wikitextDataset(test_path, max_len)
    # design net & loss
    num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, seq_len = 2, 2, 128, 0.1, 256, max_len
    net = BERT(len(trainset.vocab), num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, seq_len)
    class bertLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss = nn.CrossEntropyLoss(reduction='none')
        def forward(self, mlm_Y_hat, nsp_Y_hat, *loss_inputs_batch):
            # loss_inputs_batch = mlm_weights(bs, num_masks), mlm_labels_idx(bs, num_masks), nsp_labels(bs,)
            mlm_weights, mlm_labels_idx, nsp_labels = loss_inputs_batch
            mlm_l = self.loss(mlm_Y_hat.permute(0,2,1), mlm_labels_idx) #(bs, num_masks)
            mlm_l = (mlm_l * mlm_weights).sum(dim=1)/(mlm_weights.sum(dim=1)) #(bs,), 平均每个mask token的celoss
            nsp_l = self.loss(nsp_Y_hat, nsp_labels) #(bs,), 一次上下句判断的celoss
            l = mlm_l + nsp_l #(bs,)
            return l, mlm_l, nsp_l
    loss = bertLoss()
    # init trainer
    num_epochs, batch_size, lr = 300, 512, 0.00015
    trainer = bertTrainer(net, loss, num_epochs, batch_size)
    trainer.set_device(torch.device('cuda'))## set the device
    trainer.set_data_iter(trainset, None, testset)## set the data iters
    trainer.set_optimizer(lr)## set the optimizer
    trainer.set_grad_clipping(grad_clip_val=1.0)## set the grad clipper
    trainer.set_epoch_eval(bertEpochEvaluator(num_epochs, 'train_logs.txt', visualizer=False))## set the epoch evaluator
    # start
    trainer.log_topology('lazy_topo.txt')## print the lazy topology
    check_flag = trainer.resolve_net(need_resolve=True)## check the net & loss
    if check_flag:
        trainer.log_topology('def_topo.txt')## print the defined topology
    # fit
    trainer.fit()
    # save
    trainer.save_model('bert_test.params')

def infer_job():
    pass