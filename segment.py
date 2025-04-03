# segment.py
from Code.Utils.Text.BytePairEncoding import get_BPE_symbols, segment_word_BPE_greedy
import yaml
import os
import json




configs = yaml.load(open('Code/projs/transformer/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################
symbols_dir = os.path.join( configs['cache_dir'], configs['proj_name'], 'symbols' )




def get_corpus():

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

    return eng_corpus, fra_corpus





def save_symbols(corpus, vocab_size, need_lower, path,
                 tail_token = "</w>", reserved_tokens = [], min_occur_freq_merge = 0, normalize_whitespace = True):

    # create symbols
    symbols = get_BPE_symbols(
        text = corpus,
        tail_token = tail_token,
        merge_times = vocab_size,
        merge_mode = 'first',
        min_occur_freq_merge = min_occur_freq_merge,
        reserved_tokens = reserved_tokens,
        symbols_type = 'list',
        need_lower = need_lower,
        separate_puncs = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
        normalize_whitespace = normalize_whitespace,
        attach_tail_token_init = False
        )
    
    # 当 symbols 文件不存在时, 保存 symbols 到 path
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump(symbols, f)
    
    return len(symbols)






if __name__ == "__main__":
    eng_corpus, fra_corpus = get_corpus()

    src_vocab_size = save_symbols(
        corpus = eng_corpus,
        vocab_size = 8000,
        need_lower = True,
        path = os.path.join(symbols_dir, 'source.json')
        )

    tgt_vocab_size = save_symbols(
        corpus = fra_corpus,
        vocab_size = 8000,
        need_lower = True,
        path = os.path.join(symbols_dir, 'target.json')
        )