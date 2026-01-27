import os
import warnings
warnings.filterwarnings("ignore")
import torch
import typing as t
import pandas as pd
from .dataset import wikitextDataset
from .pretrain import bert_pretrain, bert_pretrain_loss
from .trainer import bertPreTrainer
from .evaluator import bertEpochEvaluator
from .predictor import tokensEncoder
import yaml
from src.core.models.bert import bertConfig, BERTEncoder
from src.utils.text.vocabulize import Vocab
from src.utils.text.glossary import get_BPE_glossary
from src.utils.math import cosine_similarity

configs = yaml.load(open('src/projs/bert/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## symbols and vocabs in workspace/cache ##################
glossary_dir = os.path.join( configs['artifact_dir'], configs['proj_name'], 'glossary' )
vocab_dir = os.path.join( configs['artifact_dir'], configs['proj_name'], 'vocab' )

################## params saved in workspace/model ##################
model_dir = os.path.join( configs['model_dir'], configs['proj_name'] )

################## log file in workspace/logs ##################
log_dir = os.path.join( configs['log_dir'], configs['proj_name'] )

################## train-params ##################
num_epochs, batch_size, lr = 20, 512, 0.00015




# 生产 symbols 和 vocab
def prepare_job():
    print('prepare job begin')

    # create all related directories if not existed
    for dir_name in [glossary_dir, vocab_dir, model_dir, log_dir]:
        os.makedirs(dir_name, exist_ok=True)
        print(f'directory {dir_name} created')

    # generate glossary of source/target language and save them

    # 读取全部语料 corpus
    train_df = pd.read_parquet(configs['train_data'])
    valid_df = pd.read_parquet(configs['valid_data'])
    test_df = pd.read_parquet(configs['test_data'])
    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    del train_df, valid_df, test_df
    
    corpus = ' '.join( full_df['text'].tolist() )
    uniq_size = len( set(corpus.split(' ')) )
    print(f'unique word size:{uniq_size}')

    # glossary
    # 当 glossary_path 文件不存在时, 生产并保存 wiki_glossary 到 glossary_dir/wiki.json
    glossary_path = os.path.join(glossary_dir, 'wiki.json')
    if not os.path.exists(glossary_path):
        # create wiki glossary
        glossary = get_BPE_glossary(corpus = corpus, EOW_token = "</w>", merge_times = 40000, save_path = glossary_path)

    # vocab
    # 当 vocab_path 文件不存在时, 生产并保存 wiki_vocab 到 vocab_dir/wiki.json
    vocab_path = os.path.join(vocab_dir, 'wiki.json')
    if not os.path.exists(vocab_path):
        vocab = Vocab(corpus, glossary, need_lower=True, reserved_subwords=['<pad>', '<mask>', '<cls>', '<sep>'],
                      unk='<unk>', separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|', min_freq=5)
        vocab.save(vocab_path)

    print('prepare job complete')
    return vocab_path


def pretrain_job(vocab_path):
    print('train job begin')
    # [timetag]
    from datetime import datetime
    now_minute = datetime.now().strftime("%Y-%m-%d_%H:%M")

    # /workspace/logs/[proj_name]/train_log_[timetag].txt, defined_net_[timetag].txt
    train_logs_fpath = os.path.join( log_dir, f'train_log_{now_minute}.txt' )
    defined_net_fpath = os.path.join( log_dir, f'defined_net_{now_minute}.txt' )

    # /workspace/model/[proj_name]/saved_params[timetag].params
    saved_params_fpath = os.path.join( model_dir, f'saved_params_{now_minute}.pth' )
    seq_len = configs['bertconfig'].seq_len
    trainset = wikitextDataset(configs['train_data'], vocab_path, seq_len, '<cls>', '<sep>', '<mask>')
    validset = wikitextDataset(configs['valid_data'], vocab_path, seq_len, '<cls>', '<sep>', '<mask>')
    testset = wikitextDataset(configs['test_data'], vocab_path, seq_len, '<cls>', '<sep>', '<mask>')

    # construct model
    net_config = bertConfig(vocab_size=len(testset.vocab), **configs['bertconfig'])
    net = bert_pretrain(net_config)
    # loss for train task
    loss = bert_pretrain_loss()
    # init trainer
    trainer = bertPreTrainer(net, loss, num_epochs, batch_size)
    trainer.set_device(torch.device('cuda')) # set the device
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


def embed_job(saved_params_fpath, vocab_path):
    print('embedding job begin')

    # load vocabs
    vocab = Vocab()
    vocab.load(vocab_path)

    # set device
    device = torch.device('cpu')
  
    # construct model
    net_config = bertConfig(vocab_size=len(vocab), **configs['bertconfig'])
    net = BERTEncoder(net_config).to(device)

    # load params
    net.load_state_dict(torch.load(saved_params_fpath, map_location=device))
    net.eval()
    bertEncoder = tokensEncoder(vocab, net, configs['bertconfig'].seq_len, device)
    tokens = ['a', 'crane', 'is', 'flying']
    embd_output1 = bertEncoder.predict(tokens)
    print('embedding output 1 shape: ', embd_output1.shape) # (4, num_hiddens)

    tokens, tokens_next = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
    embd_output2 = bertEncoder.predict(tokens, tokens_next)
    print('embedding output 2 shape: ', embd_output2.shape) # (4, num_hiddens)

    print('similarity for word "a":', cosine_similarity(embd_output1[0], embd_output2[0]))
    print('similarity for word "crane":', cosine_similarity(embd_output1[1], embd_output2[1]))