import os
import warnings
warnings.filterwarnings("ignore")
import torch
import typing as t
import pandas as pd
import yaml
from ...core.utils.text.vocabulize import Vocab
from ...core.utils.text.tokenizer import BBPETokenizer, ENDOFTEXT


configs = yaml.load(open('src/projs/gpt/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################
# set train log file path / network resolve output path / params save path / source&targe vocabs path

################## symbols and vocabs in workspace/cache ##################
tokenizer_dir = os.path.join( configs['artifact_dir'], configs['proj_name'], 'tokenizer' )
vocab_dir = os.path.join( configs['tmp_dir'], configs['proj_name'], 'vocab' )



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
    for dir_name in [tokenizer_dir, vocab_dir, model_dir, log_dir]:
        os.makedirs(dir_name, exist_ok=True)
        print(f'directory {dir_name} created')

    # generate tokenizer
    # not use BPE

    # 读取全部语料 corpus
    print('read train')
    train_df = pd.read_parquet(configs['train_data'])
    print('read validation')
    valid_df = pd.read_parquet(configs['valid_data'])
    print('get full corpus')
    full_df = pd.concat([train_df, valid_df], ignore_index=True)
    del train_df, valid_df
    
    corpus = ENDOFTEXT.join( full_df['text'].tolist() )

    # tokenizer
    # 当 tokenizer_path 文件不存在时, 生产并保存 tokenizer 到 tokenizer_dir/gpt.tok
    tokenizer_path = os.path.join(tokenizer_dir, 'gpt.tok')
    if not os.path.exists(tokenizer_path):
        # create tokenizer
        print('bpe train begin')
        gpt_tokenizer = BBPETokenizer(name='gpt')
        gpt_tokenizer.train_bpe(corpus, num_merges=30000, verbose=True)
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




def pretrain_job(vocab_path):
    pass