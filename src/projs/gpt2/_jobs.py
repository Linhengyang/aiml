import os
import warnings
warnings.filterwarnings("ignore")
import torch
import typing as t
import pandas as pd
import yaml
from ...core.utils.text.tokenizer import ENDOFTEXT, boostBBPETokenizer, CharacterTokenizer
from .network import gpt2Config, gpt2


configs = yaml.load(open('src/projs/gpt2/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################
# set train log file path / network resolve output path / params save path / source&targe vocabs path

################## tokenizer in workspace/artifact ##################
tokenizer_dir = os.path.join( configs['artifact_dir'], configs['proj_name'], 'tokenizer' )




################## view vocab in workspace/tmp ##################
vocab_dir = os.path.join( configs['tmp_dir'], configs['proj_name'], 'vocab' )




################## buffer directory for BPE in workspace/cache ##################
buffer_dir = os.path.join( configs['cache_dir'], configs['proj_name'], 'buffer')



################## params saved in workspace/model ##################
model_dir = os.path.join( configs['model_dir'], configs['proj_name'] )



################## log file in workspace/logs ##################
log_dir = os.path.join( configs['log_dir'], configs['proj_name'] )















################## data-params ##################
max_len = configs['max_len']





################## network-params ##################
num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias = 2, 2, 128, 0.1, 256, False






################## train-params ##################
num_epochs, batch_size, lr = 20, 512, 0.00015








# num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 4, 256, 0.1, False, 64
# batch_size
# ---------------------> 4GB 显存





# 生产 tokenizer, 可视化 view
def build_tokenizer_job():
    print('build_tokenizer_job begin')

    # create all related directories if not existed
    for dir_name in [tokenizer_dir, vocab_dir, model_dir, log_dir, buffer_dir]:
        os.makedirs(dir_name, exist_ok=True)
        print(f'directory {dir_name} created')
    
    # 读取全部语料 corpus
    print('get full corpus')
    with open(configs['train_data'], 'r', encoding='utf-8') as f:
        full_data = f.readlines() # full_data 中存在 \t \n 等其他空白符
    
    corpus = ENDOFTEXT.join(full_data)

    # tokenizer
    # 当 tokenizer_path 文件不存在时, 生产并保存 tokenizer 到 tokenizer_dir/gpt.tok
    tokenizer_path = os.path.join(tokenizer_dir, 'mt.tok')
    if not os.path.exists(tokenizer_path):
        # create tokenizer
        print('bpe train begin')
        gpt_tokenizer = boostBBPETokenizer(name='mt', buffer_dir=buffer_dir, special_marks=[ENDOFTEXT])
        gpt_tokenizer.train_bpe(30000, corpora=corpus, verbose=True)
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




def tokenize_save_job(tokenizer, corpora_paths: t.List[str], save_dir: str):
    '''
    将 corpora_paths 中的所有 txt 文件(包含EOT), tokenize 为 token IDs, 一个 text 一行, 列名为 token_id;
    每个 TEXT-DOC 从 1 开始编号(0 留给 PAD), 每行 text 都有独属的 text ID, 列名为 doc_id;
    统计每个 text 的长度(包含EOT), 为text-packing作准备, 列名为 doc_len;
    '''
    # load tokenizer. name 以及其他参数都会被 load 覆盖
    tok = boostBBPETokenizer(name='to_load', buffer_dir=buffer_dir)
    tok.load(tokenizer)

    for corpus in corpora_paths:
        fname = os.path.basename(corpus)
        save_path = os.path.join(save_dir, fname)
        
        with open(corpus, 'r', encoding='utf-8') as raw, open(save_path, 'w', encoding='utf-8') as tokenized:
            text = raw.readline()
            