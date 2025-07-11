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
    print('begin to run BPE on dataset TinyStories')
    train_pq = configs['train_pq']
    valid_pq = configs['valid_pq']
    for folder in [buffer_dir, tokenizer_save_dir, vocab_cache_dir]:
        os.makedirs(folder, exist_ok=True)
        
    tok = bufferBBPETokenizer(name='tinyTok_3', buffer_dir=buffer_dir)
    tok.train_bpe([train_pq, valid_pq], text_columns=['text', 'text'], num_merges=3, verbose=True)

    # save tokenizer
    tok_fpath = os.path.join(tokenizer_save_dir, f'{tok.name}.tok')
    tok.save(tok_fpath)
    # view vocab
    tok.view(tmpsave_dir = vocab_cache_dir)
    
    print('BPE ends')
    return tok_fpath, vocab_cache_dir




def bpe_continue(continue_num_merges):
    print('continue to run BPE on dataset TinyStories')
    tok = bufferBBPETokenizer(name='tinyTok_3', buffer_dir=buffer_dir)
    # load tokenizer
    tok.load(os.path.join(tokenizer_save_dir, 'tinyTok_3.tok'))

    # continue to train 1 epoch
    buffer_tokens_dir = os.path.join(buffer_dir, 'tokens')
    latest = max( [int(f) for f in os.listdir(buffer_tokens_dir)] )
    tokens_dir = os.path.join(buffer_tokens_dir, f'{latest}')

    tok.continue_bpe(tokens_dir, continue_num_merges)

    # save/view the new tokenizer in different name
    tok.name = 'tinyTok_4.tok'
    tok_fpath = os.path.join(tokenizer_save_dir, f'{tok.name}.tok')
    tok.save(tok_fpath)
    # view vocab
    tok.view(tmpsave_dir = vocab_cache_dir)

    print('f{continue_num_merges} BPE ends')
    return tok_fpath, vocab_cache_dir