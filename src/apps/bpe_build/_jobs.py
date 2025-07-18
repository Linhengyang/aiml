import yaml
import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
import os
import regex as re
from ...core.utils.text.tokenizer import baseBBPETokenizer, bufferBBPETokenizer, boostBBPETokenizer

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











def bpe_train():
    print('begin to run BPE on dataset TinyStories')
    train_pq = configs['train_pq']
    valid_pq = configs['valid_pq']
    for folder in [buffer_dir, tokenizer_save_dir, vocab_cache_dir]:
        os.makedirs(folder, exist_ok=True)
        
    tok = boostBBPETokenizer(name='tinyTok', buffer_dir=buffer_dir)
    corpora = [valid_pq,]
    colnames=['text',]

    tok.train_bpe(0,
                  corpora=corpora,
                  colnames=colnames,
                  backup_init_tokens_dir=backup_init_dir
                  )

    # save tokenizer
    tok_fpath = os.path.join(tokenizer_save_dir, f'{tok.name}.tok')
    tok.save(tok_fpath)
    # view vocab
    tok.view(tmpsave_dir = vocab_cache_dir)
    
    print('BPE ends')





def bpe_continue():
    print('continue to run BPE on dataset TinyStories')

    train_pq = configs['train_pq']
    valid_pq = configs['valid_pq']
    for folder in [buffer_dir, tokenizer_save_dir, vocab_cache_dir]:
        os.makedirs(folder, exist_ok=True)
    
    corpora = [valid_pq,]
    colnames=['text',]

    tok = boostBBPETokenizer(name='tinyTok', buffer_dir=buffer_dir)
    tok_path = os.path.join(tokenizer_save_dir, f'tinyTok.tok')
    tok.load(tok_path)

    tok.train_bpe(4,
                  corpora=corpora,
                  colnames=colnames,
                  backup_init_tokens_dir=backup_init_dir)
    # save tokenizer
    tok_fpath = os.path.join(tokenizer_save_dir, f'{tok.name}_new.tok')
    tok.save(tok_fpath)
    
    print('BPE ends')