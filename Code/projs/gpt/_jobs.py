import os
import warnings
warnings.filterwarnings("ignore")
import torch
import typing as t
import pandas as pd
from .Dataset import wikitextDataset
from .Network import BERT, BERTLoss
from .Trainer import bertPreTrainer
from .Evaluator import bertEpochEvaluator
from .Predictor import tokensEncoder
import yaml
from ...Utils.Text.Vocabulize import Vocab
from ...Utils.Text.Glossary import get_BPE_glossary
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
num_epochs, batch_size, lr = 20, 512, 0.00015








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
        vocab = Vocab(corpus=corpus, glossary=glossary, need_lower=True, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'],
                      unk_token='<unk>', separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|', min_freq=5)
        vocab.save(vocab_path)

    print('prepare job complete')
    return vocab_path




def pretrain_job(vocab_path):
    pass