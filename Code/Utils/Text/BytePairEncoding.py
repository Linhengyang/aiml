# BytePairEncoding.py
import collections
import typing as t
import re
from .TextPreprocess import count_corpus, preprocess_appdtokn_b4_space, text_atomize



# a learner, which learns from corpus to induce a vocaburary
# algorithm summary:
# function byte-pair-encoding(corpus C, numbers of merges k) --> vocab V
# init V <-- all unique characters in C
# repeat k times:
#   tok_L, tok_R <-- most frequent pair of adjacent tokens in C
#   tok_new <-- tok_L + tok_R
#   V <-- V + tok_new
#   update corpus: replace all occurrence of tok_L, tok_R in C with tok_new
# return V

stop_token: str = '</w>'
reserved_tokens: t.List[str ]= ['<unk>'] + [stop_token]

text = "low low lower lower   \n  lower high higher\thigh high high"
text = preprocess_appdtokn_b4_space(text, stop_token)
# text: low</w> low</w> lower</w> lower</w> lower</w> high</w> higher</w> high</w> high</w> high</w>

raw_corpus = count_corpus(text.split(" "))
symbols = set(text) | set(reserved_tokens)


# corpus C 应该是个 counter, tokcombo_freq, key 是 token combo(用空格组合 token), value 是 token组成的词对应的 freq
# corpus 会更新，要区分tokens(按照 vocab V拆分tokens)
# 所以就是对 counter的 keys 作更改，即 应该合并的tokens 合并成新 token. 但是, 因为要对 每个能改的 key 都作更改，所以需要循环。而在循环中，dict的key是不能变动的
# 两个解决办法：一 每次都生成一个新的dict，把原来的dict的key-value对 都搬到新dict
# 二 counter使用 list 数据类型. list的元素是(token_combo, freq)的元组

def init_tokcombo_freqs(raw_corpus:dict, type: str = 'list'):
    '''
    初始化一个 token combo frequency counter, 类型可以是 dict/list
    input：
        原始的 corpus 统计器 raw_corpus：一个 统计 word/punc 频数 的 dict
        类型 type：返回的 token combo 频数 corpus 的类型，dict / list
    return:
        一个  token combo 频数 统计器，类型是 type
    explain:
        token combo 频数统计器中，word/punc 被 单空格拆成了独立字符，用以迭代组合，来创造新的token
    '''
    if type == "dict":
        tokcombo_freqs = {}
        for raw_word, freq in raw_corpus.items():
            tokcombo_freqs[' '.join(text_atomize(raw_word, reserved_tokens))] = freq
        
    elif type == "list":
        tokcombo_freqs = []
        for raw_word, freq in raw_corpus.items():
            tokcombo_freqs.append( (' '.join(text_atomize(raw_word, reserved_tokens)), freq) )

    else:
        raise NotImplementedError(f'wrong type with {type}. must be one of dict/list')
    
    return tokcombo_freqs




def merge_maxfreq_token_pair(
        maxfreq_token_pairs: t.List[t.Tuple[str]],
        tokcombo_freqs: t.List[t.Tuple]|t.Dict,
        symbols:t.List|t.Set,
        merge_mode: str = 'first'):
    '''
    input
        maxfreq_token_pairs:
            list of tuples of most frequent adjacent token pair [(tok_L, tok_R), ...]

            if only one token pair has highest frequency, then it will be a list of length 1
        tokcombo_freqs:
            Dict: { token_combo_by_space: word_frequency ... } / list of tuple: [ (token_combo_by_space, word_frequency) ... ]
        symbols:
            set/list
        merge_mode: str, one of first/all/shortest ...
    output:
        updated tokcombo_freqs & symbols
    
    explains:
        tokcombo_freqs: tokcombo中, 最频繁出现的 连续 token pair 被合并. 如果 merge_mode 选择了all, 那么所有 token_pair将以 它们在 输入列表中顺序作合并
        symbols: 最频繁出现的 连续 token pair 被合并后, 添加入symbols
    '''
    if merge_mode != 'all':

        if merge_mode == 'first':
            select = 0
        elif merge_mode == 'shortest':
            tokns_len = [ len(''.join(token_pair)) for token_pair in maxfreq_token_pairs]
            select, _ = min( enumerate(tokns_len), key=lambda x:x[1] )
        else:
            raise NotImplementedError(f'merge mode {merge_mode} not implemented')
    
        maxfreq_token_pairs = maxfreq_token_pairs[select:select+1]

    # update vocab
    for token_pair in maxfreq_token_pairs:
        if isinstance(symbols, set):
            symbols.union( ''.join(token_pair) )
        elif isinstance(symbols, list):
            symbols.append( ''.join(token_pair) )
    
    # update token combo frequency corpus counter
    if isinstance(tokcombo_freqs, dict):
        new_tokcombo_freqs = {}

        for token_combo, freq in tokcombo_freqs.items():
            # 对于 token_combo / freq 这个 kv对, 不必检测 token combo 是否需要合并，因为都要搬到 新 dict里
            # 如果该 token_combo 存在 maxfreq token pair, 合并 所有maxfreq_token_pair，即去掉中间的空格; 如果不存在, 那么保持原样
            for token_pair in maxfreq_token_pairs:
                token_combo = token_combo.replace(" ".join(token_pair), "".join(token_pair))
            new_tokcombo_freqs[token_combo] = freq

        return new_tokcombo_freqs, symbols
    
    elif isinstance(tokcombo_freqs, list):

        for i, (token_combo, freq) in enumerate(tokcombo_freqs):
            # 对于 token_combo / freq 这个 kv对，需要检测 token combo 是否需要合并。因为不需要合并的不用改
            maxfreq_toknpair_pattern = '|'.join( [' '.join(token_pair) for token_pair in maxfreq_token_pairs] ) # tk1 tk2|...|tk3 tk4

            if re.match(maxfreq_toknpair_pattern, token_combo): # 如果匹配到 任意一个 maxfreq token pair
                for token_pair in maxfreq_token_pairs:
                    token_combo = token_combo.replace(" ".join(token_pair), "".join(token_pair))
                tokcombo_freqs[i] = (token_combo, freq)

        return tokcombo_freqs, symbols
    
    else:
        raise TypeError(f"wrong type for param tokcombo_freqs. must be list of tuples / dict")





def get_maxfreq_token_pair(
        tokcombo_freqs: t.List[t.Tuple]|t.Dict
        ):
    """
    input
        tokcombo_freqs:
            Dict: { token_combo_by_space: word_frequency ... } / list of tuple: [ (token_combo_by_space, word_frequency) ... ]
    return:
        token_pairs_w_maxfreq:
            list of tuples of most frequent adjacent token pair [(tok_L, tok_R), ...]
            if only one token pair has highest frequency, then it will be a list of length 1
        freq: the max frequency
    explains:
        计算 adjacent token pair 的 frequency，并返回 max freq 的 adjacent token pairs, 同时返回这个 maxfreq
    """ 
    token_pair_freq = collections.defaultdict(int)
  
    if isinstance(tokcombo_freqs, dict):
        for tokcombo, freq in tokcombo_freqs.items():
            tokens = tokcombo.split()
            num_tok = len(tokens) # num_tok > 1, OK; num_toke = 1时，说明该 token 没有merge的可能了.
            # 两种可能：一该 不能 merge 的token 是单个字符，那么最开始就会记录在 vocab 里
            # 二该 不能 merge 的token是 merge的结果. 那么 merge 的过程就会记录 该token到 vocab 里
            # 故 若 num_tok = 1，直接不跑 for chunk 是 ok 的
            for i in range(num_tok-1):
                token_pair_freq[(tokens[i], tokens[i+1])] += freq

    elif isinstance(tokcombo_freqs, list):
        for tokcombo, freq in tokcombo_freqs:
            tokens = tokcombo.split()
            num_tok = len(tokens)
            for i in range(num_tok-1):
                token_pair_freq[(tokens[i], tokens[i+1])] += freq

    # max-freq 可能有 多个 pair 达到
    maxfreq = max(token_pair_freq.values())
    token_pairs_w_maxfreq = [k for k, v in token_pair_freq.items() if v == maxfreq]

    return token_pairs_w_maxfreq, maxfreq



# a segmenter, which tokenizes a raw sentences