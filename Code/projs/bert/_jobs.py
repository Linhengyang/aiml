import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from .Dataset import wikitextDataset
from .Network import BERT, BERTLoss
from .Trainer import bertPreTrainer
from .Evaluator import bertEpochEvaluator
from .Predictor import tokensEncoder
import yaml

configs = yaml.load(open('Code/projs/bert/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################
# set train log file path / network resolve output path / params save path / source&targe vocabs path

################## symbols and vocabs in workspace/cache ##################
symbols_dir = os.path.join( configs['cache_dir'], configs['proj_name'], 'symbols' )
vocabs_dir = os.path.join( configs['cache_dir'], configs['proj_name'], 'vocabs' )


################## params saved in workspace/model ##################
model_proj_dir = os.path.join( configs['model_dir'], configs['proj_name'] )


################## log file in workspace/logs ##################
log_proj_dir = os.path.join( configs['log_dir'], configs['proj_name'] )





################## data-params ##################
max_len = configs['max_len']





################## network-params ##################
num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 4, 256, 0.1, False, 64







################## train-params ##################
num_epochs, batch_size, lr = 5, 512, 0.00005





# 生产 source corpus 和 target corpus 的symbols
def prepare_job():
    print('prepare job begin')

    # create all related directories if not existed
    for dir_name in [symbols_dir, vocabs_dir, model_proj_dir, log_proj_dir]:
        os.makedirs(dir_name, exist_ok=True)
        print(f'directory {dir_name} created')







def pretrain_job():
    print('train job begin')
    # [timetag]
    from datetime import datetime
    now_minute = datetime.now().strftime("%Y-%m-%d_%H:%M")

    # /workspace/logs/[proj_name]/train_log_[timetag].txt, defined_net_[timetag].txt
    train_logs_fpath = os.path.join( log_proj_dir, f'train_log_{now_minute}.txt' )
    defined_net_fpath = os.path.join( log_proj_dir, f'defined_net_{now_minute}.txt' )

    # /workspace/model/[proj_name]/saved_params[timetag].params
    saved_params_fpath = os.path.join( model_proj_dir, f'saved_params_{now_minute}.pth' )

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


