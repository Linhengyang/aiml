import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from .Dataset import wikitextDataset
from .Network import BERT
from .Trainer import bertPreTrainer
from .Evaluator import bertEpochEvaluator
from .Predictor import tokensEncoder
import yaml
configs = yaml.load(open('Code/projs/bert/configs.yaml', 'rb'), Loader=yaml.FullLoader)
local_model_save_dir = configs['local_model_save_dir']
base_data_dir = configs['base_data_dir']
bert_fname = configs['bert_fname']
train_fname = configs['train_fname']
test_fname = configs['test_fname']

def pretrain_job():
    train_path, max_len = os.path.join(base_data_dir, bert_fname, train_fname), 64
    test_path = os.path.join(base_data_dir, bert_fname, test_fname)
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
    trainer = bertPreTrainer(net, loss, num_epochs, batch_size)
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

def embed_job():
    device = torch.device('cpu')
    train_path, max_len = os.path.join(base_data_dir, bert_fname, train_fname), 64
    trainset = wikitextDataset(train_path, max_len)
    # design net & loss
    num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, seq_len = 2, 2, 128, 0.1, 256, max_len
    net = BERT(len(trainset.vocab), num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, seq_len)
    trained_net_path = os.path.join(local_model_save_dir, 'bert', 'bert_test.params')
    net.load_state_dict(torch.load(trained_net_path, map_location=device))
    net.eval()
    tokens_a = ['a', 'crane', 'is', 'flying']
    bertEncoder = tokensEncoder(max_len=max_len)
    embd_res = bertEncoder.predict(net, trainset.vocab, tokens_a)
    print('embedding shape: ', embd_res.shape)
    first_embd = embd_res[1]
    print('bert embedding(first 3 dims) for word "a": ', first_embd[:3])
    tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
    embd_res = bertEncoder.predict(net, trainset.vocab, tokens_a)
    print('embedding shape: ', embd_res.shape)
    second_embd = embd_res[1]
    print('bert embedding(first 3 dims) for word "a": ', second_embd[:3])
    print('similarity: ', (first_embd*second_embd).sum()/torch.sqrt((first_embd**2).sum()*(second_embd**2).sum()))