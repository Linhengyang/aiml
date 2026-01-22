import os
import warnings
warnings.filterwarnings("ignore")
import torch
import typing as t
import pandas as pd
import yaml
from src.utils.text.tokenizer import ENDOFTEXT, mpbufferBBPE_u16Tokenizer
from .network import gpt2Config, gpt2
from .loss import gpt2_pretrain_loss
from .dataset import mtDataset
from .trainer import gpt2Trainer
from .evaluator import gpt2EpochEvaluator

configs = yaml.safe_load(open('src/projs/gpt2/configs.yaml', 'rb'))

################## directories ##################
# set train log file path / network resolve output path / params save path / source&targe vocabs path

################## tokenizer in workspace/artifact ##################
tokenizer_dir = os.path.join( configs['artifact_dir'], configs['proj_name'], 'tokenizer' )

################## tokenizer in workspace/artifact ##################
data_dir = os.path.join( configs['artifact_dir'], configs['proj_name'], 'data' )

################## view vocab in workspace/tmp ##################
vocab_dir = os.path.join( configs['tmp_dir'], configs['proj_name'], 'vocab' )

################## buffer directory for BPE in workspace/cache ##################
buffer_dir = os.path.join( configs['cache_dir'], configs['proj_name'], 'buffer')

################## model params saved in workspace/model ##################
model_dir = os.path.join( configs['model_dir'], configs['proj_name'] )

################## log file in workspace/logs ##################
log_dir = os.path.join( configs['log_dir'], configs['proj_name'] )

################## train-params ##################
num_epochs, batch_size, lr = 20, 128, 0.00015


def env_set():
    # create all related directories if not existed
    for dir_name in [tokenizer_dir, vocab_dir, data_dir, buffer_dir, model_dir, log_dir]:
        os.makedirs(dir_name, exist_ok=True)
        print(f'directory {dir_name} created')


# 生产 tokenizer, 可视化 view
def build_tokenizer_job():
    print('build_tokenizer_job begin')
    
    # 读取全部语料 corpus
    print('get full corpus')
    with open(configs['train_data'], 'r', encoding='utf-8') as f:
        full_data = f.readlines() # full_data 中存在 \t \n 等其他空白符
    
    corpus = ENDOFTEXT.join(full_data)

    # tokenizer
    # 当 tokenizer_path 文件不存在时, 生产并保存 tokenizer 到 tokenizer_dir/gpt.tok
    tokenizer_path = os.path.join(tokenizer_dir, 'mt.tok')

    num_merges, specials = 30000, [ENDOFTEXT]
    if not os.path.exists(tokenizer_path):
        # create tokenizer
        print('bpe train begin')
        gpt_tokenizer = mpbufferBBPE_u16Tokenizer(name='mt', buffer_dir=buffer_dir, special_marks=specials)
        gpt_tokenizer.train_bpe(num_merges, corpora=corpus, verbose=True)
        print('bpe train close')
        gpt_tokenizer.save(tokenizer_path)
    # 当 tokenizer_path 存在时
    else:
        gpt_tokenizer.load(tokenizer_path)

    # vocab
    print('bpe view')
    gpt_tokenizer.view(vocab_dir)

    print('build_tokenizer_job complete')
    return tokenizer_path


@torch.no_grad()
def tokenize_corpus(mt_text_path, tok_path, data_path):

    with open(mt_text_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    tok = mpbufferBBPE_u16Tokenizer(name='.', buffer_dir='.')
    tok.load(tok_path)

    lines = raw_text.split('\n')
    data = torch.empty(2,0, dtype=torch.long) # 2 for tokens/segments

    for i, line in enumerate(lines):
        print(f'tokenize line {i}')
        line += ENDOFTEXT # str append
        tokens = tok.encode(line, allowed_special=set([ENDOFTEXT])) # str tokenize to list of ints
        segments = [i+1]*len(tokens)
        datapoint = torch.tensor([tokens, segments], dtype=torch.long) # [2, l]
        data = torch.concat([data, datapoint], dim=-1) # [2, L + l]

    torch.save(data, data_path)



def pretrain_job():
    print('train job begin')
    # [timetag]
    from datetime import datetime
    now_minute = datetime.now().strftime("%Y-%m-%d_%H:%M")

    # /workspace/logs/[proj_name]/train_log_[timetag].txt
    train_log_path = os.path.join( log_dir, f'train_log_{now_minute}.txt' )

    # /workspace/model/[proj_name]/saved_params[timetag].params
    saved_params_path = os.path.join(model_dir, f'saved_params_{now_minute}.pth')

    tok_path = os.path.join(tokenizer_dir, 'mt.tok')
    data_path = os.path.join(data_dir, 'mt_data.pt')

    if not os.path.exists(data_path):
        tokenize_corpus(configs['train_data'], tok_path, data_path)

    trainset = mtDataset(data_path, configs['seq_len'])

    # design net & loss
    net_config = gpt2Config(**configs['gpt2config'])
    net = gpt2(net_config)
    loss = gpt2_pretrain_loss()

    # init trainer
    trainer = gpt2Trainer(net, loss, num_epochs, batch_size)

    trainer.set_device(torch.device('cuda')) # set the device
    trainer.set_data_iter(trainset) # set the data iters
    trainer.set_optimizer(lr) # set the optimizer
    trainer.set_grad_clipping(grad_clip_val=1.0) # set the grad clipper

    # fit model
    evaluator = gpt2EpochEvaluator(num_epochs, train_log_path, verbose=True)
    trainer.fit(evaluator)
    
    # save
    trainer.save_model(saved_params_path)

    print('pretrain job complete')
    return saved_params_path


def generate_job():
    device = torch.device('cuda')
    saved_params_path = os.path.join(model_dir, f'saved_params_2025-09-30_12:04.pth')
    net_config = gpt2Config(**configs['gpt2config'])
    net = gpt2(net_config).to(device)

    net.load_state_dict(torch.load(saved_params_path, map_location=device))

    tok_path = os.path.join(tokenizer_dir, 'mt.tok')
    tok = mpbufferBBPE_u16Tokenizer(name='.', buffer_dir='.')
    tok.load(tok_path)

    prompt = tok.encode("I don't mind it.	")
    segments = [1]*len(prompt)

    prompt_ts = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    init_segs_ts = torch.tensor(segments, dtype=torch.long, device=device).unsqueeze(0)

    answer = net.generate(prompt_ts, init_segs_ts, 10).squeeze(0) # [10, ]

    print(tok.decode(answer.tolist()))
