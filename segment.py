# segment.py
from Code.Utils.Text.BytePairEncoding import get_BPE_symbols, segment_word_BPE_greedy
import yaml
import os
import json




configs = yaml.load(open('Code/projs/transformer/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################
symbols_dir = os.path.join( configs['cache_dir'], configs['proj_name'], 'symbols' )




# 读取全部语料 corpus
with open(configs['full_data'], 'r', encoding='utf-8') as f:
    full_data = f.readlines() # full_data 中存在 \t \n 等其他空白符


# 分割成 english 和 france. 由于本次任务对 source language 和 target language 采用 分开独立的词汇表 和 嵌入空间, 故 BPE 生成 symbols 也是独立的
eng_corpus, fra_corpus = [], []

for line in full_data:
    eng_sentence, fra_sentence = line.split('\t')
    eng_corpus.append(eng_sentence)
    fra_corpus.append(fra_sentence)


eng_corpus = " ".join(eng_corpus)
fra_corpus = " ".join(fra_corpus)





eng_symbols = get_BPE_symbols(
    text = eng_corpus,
    tail_token = "</w>",
    merge_times = 10000,
    merge_mode = 'first',
    min_occur_freq_merge = 0,
    reserved_tokens = [],
    symbols_type = 'list',
    need_lower = True,
    separate_puncs = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
    normalize_whitespace = True, # 把 \t 和 \n 替换成单空格
    attach_tail_token_init = False
    )

with open(os.path.join(symbols_dir, 'source.json'), 'w') as f:
    json.dump(eng_symbols, f)




fra_symbols = get_BPE_symbols(
    text = fra_corpus,
    tail_token = "</w>",
    merge_times = 10000,
    merge_mode = 'first',
    min_occur_freq_merge = 0,
    reserved_tokens = [],
    symbols_type = 'list',
    need_lower = True,
    separate_puncs = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
    normalize_whitespace = True, # 把 \t 和 \n 替换成单空格
    attach_tail_token_init = False
    )

with open(os.path.join(symbols_dir, 'target.json'), 'w') as f:
    json.dump(fra_symbols, f)