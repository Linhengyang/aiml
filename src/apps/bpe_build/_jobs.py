import yaml
import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
import os
import regex as re
from ...core.utils.text.tokenizer import baseBBPETokenizer, bufferBBPETokenizer, boostBBPETokenizer, mpBBPETokenizer

configs = yaml.load(open('src/apps/bpe_build/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################


################## buffer/save in workspace/cache ##################
buffer_dir = os.path.join( configs['cache_dir'], configs['app_name'], 'buffer' )
tokenizer_save_dir = os.path.join( configs['artifact_dir'], configs['app_name'], 'tokenizer' )
vocab_cache_dir = os.path.join( configs['artifact_dir'], configs['app_name'], 'vocab' )


################## backup init to data/TinyStories/bytes ##################
backup_init_dir = configs['backup_init_dir']



################## backup init to data/TinyStories/bytes ##################
num_merges = configs['num_merges']
buffer_size = configs['buffer_size']
save_tok_name = configs['save_tok_name']









def bpe_prepare():
    print('insight on dataset TinyStories')
    train_pq = configs['train_pq']
    valid_pq = configs['valid_pq']

    for pq_file in [train_pq, valid_pq]:
        meta_data = pq.ParquetFile(pq_file).metadata
        print(f'parquet file {pq_file} info:\n{pq.ParquetFile(pq_file).metadata}')

        if meta_data.num_rows // buffer_size >= 200:
            raise RuntimeError(f'Num of Batches {meta_data.num_rows // buffer_size} exceeds 200.')











def bpe_train():
    print('begin to run BPE on dataset TinyStories')
    train_pq = configs['train_pq']
    valid_pq = configs['valid_pq']
    for folder in [buffer_dir, tokenizer_save_dir, vocab_cache_dir]:
        os.makedirs(folder, exist_ok=True)
        
    tok = mpBBPETokenizer(name=save_tok_name, buffer_dir=buffer_dir)
    corpora = [valid_pq, ]
    colnames = ['text']*len(corpora)

    tok.train_bpe(num_merges,
                  corpora=corpora,
                  colnames=colnames,
                  backup_init_tokens_dir=None,
                #   buffer_size=buffer_size,
                  keep_window=0,
                  verbose=True
                  )

    # save tokenizer
    tok_fpath = os.path.join(tokenizer_save_dir, f'{tok.name}.tok')
    tok.save(tok_fpath)
    # view vocab
    tok.view(tmpsave_dir = vocab_cache_dir)
    
    print('BPE ends')





def bpe_continue(tok_path:str|None):
    print('continue to run BPE on dataset TinyStories')
    tok = mpBBPETokenizer(name='init', buffer_dir=buffer_dir)

    if tok_path and os.path.isfile(tok_path):
        tok.load(tok_path)

    # continue train 只要 buffer_dir 的环境正确, load 的tok正确, 就会自动分析该读取的中间文件，以继续训练
    # 当 tok_path 为 None 时, 相当于 load 一个只有 0-255 值映射的 tokenizer, 配合 buffer_dir/tokens/0 续train

    # continue train 下, corpora必须显式地输入None
    tok.train_bpe(num_merges,
                  corpora=None,verbose=True,
                  keep_window = 0,
                #   buffer_size=buffer_size,
                  )

    # rename and save the updated tokenizer
    tok.name = save_tok_name
    tok_fpath = os.path.join(tokenizer_save_dir, f'{save_tok_name}.tok')
    tok.save(tok_fpath)

    # view vocab
    tok.view(tmpsave_dir = vocab_cache_dir)
    print('BPE ends')