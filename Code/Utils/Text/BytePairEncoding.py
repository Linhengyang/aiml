# BytePairEncoding.py
import collections
import typing as t
import re
from .TextPreprocess import count_corpus, preprocess_space, preprocess_appdtokn_b4_space, text_atomize



# a learner, which learns from corpus to induce a vocaburary(symbols)
# algorithm summary:
# function byte-pair-encoding(corpus C, numbers of merges k) --> vocab V(symbols)
# init V <-- all unique characters in C
# repeat k times:
#   tok_L, tok_R <-- most frequent pair of adjacent tokens in C
#   tok_new <-- tok_L + tok_R
#   update vocab: V <-- V + tok_new
#   update corpus: replace all occurrence of tok_L, tok_R in C with tok_new
# return V



# corpus C 应该是个 counter, tokcombo_freq, key 是 token combo(用空格组合 token), value 是 token组成的词对应的 freq
# corpus 会更新，要区分tokens(按照 vocab(symbols) V拆分tokens)
# 所以就是对 counter的 keys 作更改，即 应该合并的tokens 合并成新 token. 但是, 因为要对 每个能改的 key 都作更改，所以需要循环。而在循环中，dict的key是不能变动的
# 两个解决办法：一 每次都生成一个新的dict，把原来的dict的key-value对 都搬到新dict
# 二 counter使用 list 数据类型. list的元素是(token_combo, freq)的元组

def init_tokcombo_freqs(
        raw_corpus:dict,
        reserved_tokens:t.List[str],
        type: str = 'list'):
    '''
    初始化一个 token combo frequency counter, 类型可以是 dict/list
    input：
        raw_corpus：原始的 corpus 统计器，一个 统计 word/punc 频数 的 dict. word/punc 按照原始方式组织
        reserved_tokens：保留字符, 作为整体不可分割的 token 列表. 在初始化分割时 就保留它们作为最小token
        tail_token：标识原词末尾的符号。它和它前面的非空字符不被分割。
        type：类型，返回的 token combo 频数 corpus 的类型，dict / list
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

    # update vocab(symbols)
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
            
            if re.search(maxfreq_toknpair_pattern, token_combo): # 如果匹配到 任意一个 maxfreq token pair
                
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
            # 两种可能：一该 不能 merge 的token 是单个字符，那么最开始就会记录在 vocab(symbols) 里
            # 二该 不能 merge 的token是 merge的结果. 那么 merge 的过程就会记录 该token到 vocab(symbols) 里
            # 故 若 num_tok = 1，直接不跑 for chunk 是 ok 的
            for i in range(num_tok-1):
                token_pair_freq[(tokens[i], tokens[i+1])] += freq
    
    elif isinstance(tokcombo_freqs, list):
        for tokcombo, freq in tokcombo_freqs:
            tokens = tokcombo.split()
            num_tok = len(tokens)
            for i in range(num_tok-1):
                token_pair_freq[(tokens[i], tokens[i+1])] += freq
    
    # 处理 token_pair_freq 为空的极端情况：当且仅当 tokcombo_freqs 为空，又或者 tokcombo_freqs 中所有 tokcombo 都是单token, 即无可合并
    if not token_pair_freq:
        return (), 0

    # max-freq 可能有 多个 pair 达到
    maxfreq = max(token_pair_freq.values())
    token_pairs_w_maxfreq = [k for k, v in token_pair_freq.items() if v == maxfreq]
    
    return token_pairs_w_maxfreq, maxfreq



def get_BPE_symbols(
        text,
        tail_token,
        merge_times: int,
        merge_mode: str = 'first',
        merge_occur_freq_min: int = 0,
        reserved_tokens: t.List[str] | None = None,
        symbols_type: str = 'list',
        need_lower: bool = True,
        separate_puncs: str = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
        normalize_whitespace: bool = True
        ) -> t.List|t.Set:
    
    # 输入文本中不应该存在用以分割的 tail_token. 如果存在, 报错;
    if tail_token in text:
        raise ValueError(f'tail_token {tail_token} exists in text. change tail_token')

    if not reserved_tokens:
        reserved_tokens = []
    
    # 最后输出的 symbols 包含 tail_token 和 输入的 reserved_tokens
    if tail_token not in reserved_tokens:
        reserved_tokens.append( tail_token )

    # 处理空白字符. 在标点前面添加空格
    text_normspace = preprocess_space(text, need_lower, separate_puncs, normalize_whitespace)

    # 在每个单词/标点末尾添加 tail_token
    text_normspace_appdtail = preprocess_appdtokn_b4_space(text_normspace, tail_token)

    raw_corpus = count_corpus(text_normspace_appdtail.split(" "))

    # 初始化 tokcombo_freqs：用单空格 分割 corpus 中的key，至不可分割粒度。
    tokcombo_freqs = init_tokcombo_freqs(raw_corpus, reserved_tokens, type='list') # 默认使用 list 来制作 tokcombo freq counter
    
    # 初始化 symbols：包含 tail_token 和 输入的 reserved_tokens 和 
    symbols = set(text_normspace) | set(reserved_tokens) # text 的 unique 单字符 union 保留字符 reserved_tokens
    
    # 这里也可以不 union reserved_tokens.
    # 因为 reversed_tokens 在合并生成 bpe symbols 的过程中没有起作用, 而且它可以在 vocab 类中设定. 故它不是必须的
    if symbols_type == 'list':
        symbols = list(symbols)

    # merge 一定次数, 或 maxfreq 的token pair 的出现频次 低于 阈值
    for _ in range(merge_times):

        token_pairs_w_maxfreq, maxfreq = get_maxfreq_token_pair(tokcombo_freqs) # 得到 max freq token pairs
        
        # 当 token pair occurrence freq >= min_freq 且 maxfreq > 0 时，才进行 merge 操作
        if maxfreq > 0 and maxfreq >= merge_occur_freq_min:
            # merge maxfreq token pair(s) : update vocab(symbols), tokcombo_freqs, 
            tokcombo_freqs, symbols = merge_maxfreq_token_pair(token_pairs_w_maxfreq, tokcombo_freqs, symbols, merge_mode)
        
    return symbols





# a segmenter, which tokenizes a raw sentences