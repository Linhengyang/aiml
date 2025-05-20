import os
import warnings
warnings.filterwarnings("ignore")
import torch
import typing as t
from .Dataset import wikitextDataset
from .Network import BERT, BERTLoss
from .Trainer import bertPreTrainer
from .Evaluator import bertEpochEvaluator
from .Predictor import tokensEncoder
import yaml
from ...Utils.Text.Vocabulize import Vocab
from ...Utils.Text.BytePairEncoding import get_BPE_glossary
from ...Utils.System.Math import cosine_similarity

configs = yaml.load(open('Code/projs/bert/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################
# set train log file path / network resolve output path / params save path / source&targe vocabs path

################## symbols and vocabs in workspace/cache ##################
glossary_dir = os.path.join( configs['cache_dir'], configs['proj_name'], 'glossary' )
vocab_dir = os.path.join( configs['cache_dir'], configs['proj_name'], 'vocab' )



################## params saved in workspace/model ##################
model_dir = os.path.join( configs['model_dir'], configs['proj_name'] )



################## log file in workspace/logs ##################
log_dir = os.path.join( configs['log_dir'], configs['proj_name'] )















################## data-params ##################
max_len = configs['max_len']





################## network-params ##################
num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias = 2, 2, 128, 0.1, 256, False






################## train-params ##################
num_epochs, batch_size, lr = 10, 512, 0.00015








# num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 4, 256, 0.1, False, 64
# batch_size
# ---------------------> 4GB 显存





# 生产 symbols 和 vocab
def prepare_job():
    print('prepare job begin')

    # create all related directories if not existed
    for dir_name in [glossary_dir, vocab_dir, model_dir, log_dir]:
        os.makedirs(dir_name, exist_ok=True)
        print(f'directory {dir_name} created')

    # generate glossary of source/target language and save them
    # not use BPE

    # 读取全部语料 corpus
    with open(configs['full_data'], 'r', encoding='utf-8') as f:
        corpus = f.read() # corpus 中存在 \t \n 等其他空白符

    uniq_size = len( set(corpus.split(' ')) )
    print(f'unique word size:{uniq_size}')

    # glossary
    # 当 glossary_path 文件不存在时, 生产并保存 wiki_glossary 到 glossary_dir/wiki.json
    glossary_path = os.path.join(glossary_dir, 'wiki.json')
    if not os.path.exists(glossary_path):
        # create wiki glossary
        glossary = get_BPE_glossary(corpus = corpus, EOW_token = "</w>", merge_times = 15000, save_path = glossary_path)


    # vocab
    # 当 vocab_path 文件不存在时, 生产并保存 wiki_vocab 到 vocab_dir/wiki.json
    vocab_path = os.path.join(vocab_dir, 'wiki.json')
    if not os.path.exists(vocab_path):
        vocab = Vocab(corpus=corpus, glossary=glossary, need_lower=True, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'],
                      unk_token='<unk>', separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|', min_freq=5)
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

    trainset = wikitextDataset(fpath=configs['train_data'], vocab_path = vocab_path, max_len = max_len,
                               cls_token = '<cls>', eos_token = '<sep>', mask_token='<mask>')
    validset = wikitextDataset(fpath=configs['valid_data'], vocab_path = vocab_path, max_len = max_len,
                               cls_token = '<cls>', eos_token = '<sep>', mask_token='<mask>')
    testset = wikitextDataset(fpath=configs['test_data'], vocab_path = vocab_path, max_len = max_len,
                              cls_token = '<cls>', eos_token = '<sep>', mask_token='<mask>')


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












def embed_job(saved_params_fpath, vocab_path):
    print('embedding job begin')

    # load vocabs
    vocab = Vocab()
    vocab.load(vocab_path)

    # set device
    device = torch.device('cpu')
  
    # construct model
    net_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blks}
    
    net = BERT(vocab_size=len(vocab), **net_args)

    # load params
    net.load_state_dict(torch.load(saved_params_fpath, map_location=device))
    net.eval()

    bertEncoder = tokensEncoder(vocab, net, max_len, device)


    tokens = ['a', 'crane', 'is', 'flying']
    embd_output1 = bertEncoder.predict(tokens)
    print('embedding output 1 shape: ', embd_output1.shape) # (4, num_hiddens)
    

    tokens, tokens_next = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
    embd_output2 = bertEncoder.predict(tokens, tokens_next)
    print('embedding output 2 shape: ', embd_output2.shape) # (4, num_hiddens)


    print('similarity for word "a":', cosine_similarity(embd_output1[0], embd_output2[0]))
    print('similarity for word "crane":', cosine_similarity(embd_output1[1], embd_output2[1]))


