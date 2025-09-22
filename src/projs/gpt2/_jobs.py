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
        gpt_tokenizer = boostBBPETokenizer(name='mt')
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




def pretrain_job(vocab_path):
    pass




def test_job():
    
    test_gpt2_cfg = gpt2Config(
        embd_size = 64,
        vocab_size = 27, # a-z <-> 1-26, <eot> <-> 0
        embd_p_drop = 0.1,
        ## decoder-block(casual_attention layer + ffn) configs
        # embd_size:int
        num_head = 2,
        use_bias = False,
        max_context_size = 6,
        attn_p_drop = 0.1,
        resid_p_drop = 0.1,
        use_cached_casual_mask = False,
        use_rope = True,
        ## number of decoder-block
        num_block = 2
    )
    chartok = CharacterTokenizer()
    gpt2_model = gpt2(test_gpt2_cfg)

    corpus = ['abcde', 'heell', 'hello']
    input_seqs = torch.tensor( [ chartok.encode(s) for s in corpus] ) - 96 # [3, 5]
    input_segs = torch.ones_like(input_seqs, dtype=torch.long)

    # 右 PAD. PAD id --> 0
    input_seqs = torch.cat([input_seqs, torch.zeros(3, 1, dtype=torch.long)], dim=1)
    input_segs = torch.cat([input_segs, torch.zeros(3, 1, dtype=torch.long)], dim=1)

    logits, _, _ = gpt2_model(input_seqs, input_segs=input_segs)

    print(logits.shape)

    # 测试 generate. eos_id 设定为 PAD id 0
    prompt = ['aa', 'ab']
    prompt_seqs = torch.tensor( [ chartok.encode(s) for s in prompt] ) - 96 # [B=2, S=2]
    prompt_segs = torch.ones_like(prompt_seqs, dtype=torch.long)

    max_gen_size = 4

    output_ids = gpt2_model.generate(prompt_seqs, prompt_segs, max_gen_size, eos_id=0) + 96 # [B=2, max_gen_size=4]
    output = []
    for seq in output_ids.tolist():
        output.append( chartok.decode(seq) )
    
    print(output)