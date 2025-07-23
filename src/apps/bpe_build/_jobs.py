import yaml
import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
import os
import regex as re
from ...core.utils.text.tokenizer import baseBBPETokenizer, bufferBBPETokenizer, boostBBPETokenizer, asyncBBPETokenizer

configs = yaml.load(open('src/apps/bpe_build/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################


################## buffer/save in workspace/cache ##################
buffer_dir = os.path.join( configs['cache_dir'], configs['app_name'], 'buffer' )
tokenizer_save_dir = os.path.join( configs['artifact_dir'], configs['app_name'], 'tokenizer' )
vocab_cache_dir = os.path.join( configs['artifact_dir'], configs['app_name'], 'vocab' )


################## backup init to data/TinyStories/bytes ##################
backup_init_dir = configs['backup_init_dir']




def bpe_prepare():
    print('insight on dataset TinyStories')
    train_pq = configs['train_pq']
    valid_pq = configs['valid_pq']
    
    for pq_file in [train_pq, valid_pq]:
        print(f'parquet file {pq_file} info:\n{pq.ParquetFile(pq_file).metadata}')











def bpe_train(num_merges:int, save_tok_name:str):
    print('begin to run BPE on dataset TinyStories')
    train_pq = configs['train_pq']
    valid_pq = configs['valid_pq']
    for folder in [buffer_dir, tokenizer_save_dir, vocab_cache_dir]:
        os.makedirs(folder, exist_ok=True)
        
    tok = asyncBBPETokenizer(name=save_tok_name, buffer_dir=buffer_dir)
    corpora = [valid_pq,]
    colnames=['text',]

    tok.train_bpe(num_merges,
                  corpora=corpora,
                  colnames=colnames,
                  backup_init_tokens_dir=backup_init_dir,
                  verbose=True
                  )

    # save tokenizer
    tok_fpath = os.path.join(tokenizer_save_dir, f'{tok.name}.tok')
    tok.save(tok_fpath)
    # view vocab
    tok.view(tmpsave_dir = vocab_cache_dir)
    
    print('BPE ends')





def bpe_continue(num_merges:int, save_tok_name:str, *, tok_path:str|None):

    print('continue to run BPE on dataset TinyStories')
    tok = asyncBBPETokenizer(name='init', buffer_dir=buffer_dir)

    if tok_path and os.path.isfile(tok_path):
        tok.load(tok_path)

    # continue train 只要 buffer_dir 的环境正确, load 的tok正确
    # 就会自动分析该读取的中间文件，以继续训练

    # continue train 下, corpora必须显式地输入None
    tok.train_bpe(num_merges, corpora=None, verbose=True)

    # rename and save the updated tokenizer
    tok.name = save_tok_name
    tok_fpath = os.path.join(tokenizer_save_dir, f'{save_tok_name}.tok')
    tok.save(tok_fpath)

    # view vocab
    tok.view(tmpsave_dir = vocab_cache_dir)
    print('BPE ends')