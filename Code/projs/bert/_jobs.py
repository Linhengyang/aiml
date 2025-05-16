import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import typing as t
from .Dataset import wikitextDataset
from .Network import BERT, BERTLoss
from .Trainer import bertPreTrainer
from .Evaluator import bertEpochEvaluator
from .Predictor import tokensEncoder
import yaml
from ...Utils.Text.BytePairEncoding import get_BPE_symbols

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
num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias = 2, 2, 128, 0.1, 256, False






################## train-params ##################
num_epochs, batch_size, lr = 300, 512, 0.00015





# 生产 symbols 和 vocab
def prepare_job():
    print('prepare job begin')

    # create all related directories if not existed
    for dir_name in [symbols_dir, vocabs_dir, model_proj_dir, log_proj_dir]:
        os.makedirs(dir_name, exist_ok=True)
        print(f'directory {dir_name} created')

    # use simple tokenizer. generate symbols and vocab directly










def pretrain_job(Datasets: t.List[torch.utils.data.Dataset]):
    print('train job begin')
    # [timetag]
    from datetime import datetime
    now_minute = datetime.now().strftime("%Y-%m-%d_%H:%M")

    # /workspace/logs/[proj_name]/train_log_[timetag].txt, defined_net_[timetag].txt
    train_logs_fpath = os.path.join( log_proj_dir, f'train_log_{now_minute}.txt' )
    defined_net_fpath = os.path.join( log_proj_dir, f'defined_net_{now_minute}.txt' )

    # /workspace/model/[proj_name]/saved_params[timetag].params
    saved_params_fpath = os.path.join( model_proj_dir, f'saved_params_{now_minute}.pth' )

    trainset, validset, testset = Datasets

    # design net & loss
    net_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blks}
    
    net = BERT(vocab_size=len(trainset.vocab), **net_args)
    loss = BERTLoss()

    # init trainer
    trainer = bertPreTrainer(net, loss, num_epochs, batch_size)

    trainer.set_device(torch.device('cpu')) # set the device
    trainer.set_data_iter(trainset, validset, testset) # set the data iters
    trainer.set_optimizer(lr) # set the optimizer
    trainer.set_grad_clipping(grad_clip_val=1.0) # set the grad clipper
    trainer.set_epoch_eval(bertEpochEvaluator(num_epochs, train_logs_fpath, verbose=True)) # set the epoch evaluator

    # set trainer
    check_flag = trainer.resolve_net(need_resolve=True)## check the net & loss
    if check_flag:
        trainer.log_topology(defined_net_fpath)## print the defined topology
        trainer.init_params()## init params

    # fit model
    trainer.fit()
    
    # save
    trainer.save_model(saved_params_fpath)

    print('train job complete')
    return saved_params_fpath


















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


