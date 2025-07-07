import yaml
import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
import os
import regex as re
from ...core.utils.text.tokenizer import baseBBPETokenizer, bufferBBPETokenizer

configs = yaml.load(open('src/apps/bpe_build/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################


################## buffer/save in workspace/cache ##################
buffer_dir = os.path.join( configs['cache_dir'], configs['app_name'], 'buffer' )
tokenizer_save_dir = os.path.join( configs['artifact_dir'], configs['app_name'], 'tokenizer' )
vocab_cache_dir = os.path.join( configs['artifact_dir'], configs['app_name'], 'vocab' )

















def bpe_build():
    print('begin to run BPE on dataset TinyStories with 50k vocab_size')
    train_pq = configs['train_pq']
    valid_pq = configs['valid_pq']

    tok = bufferBBPETokenizer(name='tinyTok_50k', buffer_dir=buffer_dir)
    tok.train_bpe(train_pq, tetext_column='text', num_merges=50000)

    # save tokenizer
    tok_fpath = os.path.join(tokenizer_save_dir, f'{tok.name}.tok')
    tok.save(tok_fpath)
    # view vocab
    tok.view(tmpsave_dir = vocab_cache_dir)
    
    print('BPE ends')
    return tok_fpath, vocab_cache_dir