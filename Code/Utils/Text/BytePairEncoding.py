# BytePairEncoding.py
import collections
import typing as t
import re
from .TextPreprocess import count_corpus, preprocess_space, attach_EOW_token, text_atomize



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
        tail_token:str = '',
        attach_tail_token_init: bool = False,
        type:str = 'list'):
    '''
    初始化一个 token combo frequency counter, 类型可以是 dict/list
    input：
        raw_corpus：原始的 corpus 统计器，一个 统计 word/punc 频数 的 dict. word/punc 按照原始方式组织
        reserved_tokens：保留字符, 作为整体不可分割的 token 列表. 在初始化分割时 保留它们作为最小token
        tail_token：标识原词末尾的符号。
        attach_tail_token_init：True/False. if True, tail_token和它前面的字符在初始化 token combos的时候 不被分割。
        type：类型，返回的 token combo 频数 corpus 的类型，可以是 dict / list
    return:
        一个  token combo 频数 统计器，类型是 输入参数 type
    explain:
        token combo 频数统计器中，word/punc 被 单空格拆成了独立字符。用以迭代组合，来创造新的token。每次迭代将出现频次最高的连续token合并，生成新token
        参数 reserved_tokens 指明了哪些 字符combo 不会被拆分, 作为独立字符
        参数 tail_token 指明了 end-of-word token。它将被加到 每个 word/punc 的末尾，以区分 中断 和 结束。tail_token 作为 独立字符，不会被拆分
        参数 attach_tail_token_init 指明了 end-of-word token 是如何被加到每个 word/punc 的末尾的。
            True: tail_token 和 末尾char 之间不分割，即 tail_token 从最开始就和末尾字符绑定。
            False: tail_token 和 末尾char 之间分割。tail_token 作为一个独立 的 字符，参与 token 的合并生成过程
            区别：业界通常使用 False，这样 token 生成的 灵活性更高，
    '''
    # 输入了 tail_token

    # 要求 tail_token和它前面的字符在初始化 token combos的时候 不分割。
    if tail_token and attach_tail_token_init:
        # 最先匹配 (.tail_token), 即完整的tail_token 和它前面一个字符
        reserved_tokens = ['.'+tail_token, ] + reserved_tokens
    
    # 要求 tail_token和它前面的字符在初始化 token combos的时候 被分割。
    elif tail_token:
        # tail_token 自身应该作为 独立字符，保证不被 分割
        reserved_tokens.append( tail_token )

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
        merge_mode: str, one of first/all/shortest/random ...
    output:
        updated tokcombo_freqs & symbols
    
    explains:
        tokcombo_freqs: tokcombo中, 最频繁出现的 连续 token pair 被合并. 如果 merge_mode 选择了all, 那么所有 token_pair将以 它们在 输入列表中顺序作合并
        symbols: 最频繁出现的 连续 token pair 被合并后, 添加入symbols
    '''
    
    if merge_mode != 'all': # 当 mode 不为 all 时，以某种方式确定 单个 token pair 以合并

        if merge_mode == 'first': # 合并 maxfreq 的 token pair 列表中的 第一对pair
            select = 0
        elif merge_mode == 'shortest': # 合并 maxfreq 的 token pair 列表中的 最短的 pair
            tokns_len = [ len(''.join(token_pair)) for token_pair in maxfreq_token_pairs]
            select, _ = min( enumerate(tokns_len), key=lambda x:x[1] )
        elif merge_mode == 'random': # 合并 maxfreq 的 token pair 列表中的 随机某个 pair
            import random
            select = random.randrange(0, len(maxfreq_token_pairs))
        else:
            raise NotImplementedError(f'merge mode {merge_mode} not implemented')
    
        maxfreq_token_pairs = maxfreq_token_pairs[select:select+1]

    # update vocab(symbols)
    for token_pair in maxfreq_token_pairs: # 逐一在 symbols 添加 合并后的 token pair 作为 新token
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
        min_occur_freq_merge: int = 0,
        reserved_tokens: t.List[str] = [],
        symbols_type: str = 'list',
        need_lower: bool = True,
        separate_puncs: str = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
        normalize_whitespace: bool = True,
        attach_tail_token_init: bool = False
        ) -> t.List|t.Set:
    '''
    input:
        text: 输入文本, 用以 构建 token symbols 集合（vocab）
        tail_token: 用来标记每个 单词的末尾，区分从中间分割的token和结尾token。会被加入到 输出的集合中。
        merge_times: 超参数，用来确定 生成的 token symbols集合大小
        merge_mode: 当有多个 token pair 是 最大出现频率的时候，采用什么方法选择 该合并的 token pair
            all全部都合并 / first第一个合并 / random 随机选 / shortest 最短的合并
        min_occur_freq_merge: 最低要合并的token pair 出现频率。出现频率小于这个值的 token pair 不再合并添加到集合中
        reserved_tokens: 保留token组合. 列表中的tokens将被作为最小单元保留, 不会被分割。会出现在 输出的集合中
        symbols_type: 输出的集合数据类型。list / set
        need_lower: 输入text是否要小写化
        separate_puncs: 作为独立视作token的标点符号
        normalize_whitespace: 是否将text中非 单空格 的连续空白字符or空白字符 统一转换为 单空格
        attach_tail_token_init: 是否绑定 tail_token 和 末尾char。如果否，tail_token 作为独立字符 参与token生成
    '''
    # 输入文本中不应该存在用以分割的 tail_token. 如果存在, 报错;
    if tail_token in text:
        raise ValueError(f'tail_token {tail_token} exists in text. change tail_token')
    
    # 处理空白字符. 在标点前面添加 单空格
    text_normspace = preprocess_space(text, need_lower, separate_puncs, normalize_whitespace)

    # 在每个单词/标点末尾添加 tail_token
    text_normspace_appdtail = attach_EOW_token(text_normspace, tail_token)

    # 原始的 corpus counter
    raw_corpus = count_corpus(text_normspace_appdtail.split(" "))

    # 初始化 tokcombo_freqs：用单空格 分割 corpus 中的key，至不可分割粒度。参数 reserved_tokens 是不可分割token
    tokcombo_freqs = init_tokcombo_freqs(raw_corpus, reserved_tokens, tail_token, attach_tail_token_init)
    
    # 初始化 symbols：包含 输入文本text的所有非空字符、单空格、输入的保留字符组合 reserved_tokens、tail_token
    symbols = set(text_normspace) | set(reserved_tokens) | set([tail_token])
    
    # 这里也可以不 union reserved_tokens.
    # 因为 reversed_tokens 在合并生成 bpe symbols 的过程中没有起作用, 而且它可以在 vocab 类中设定. 故它不是必须的
    if symbols_type == 'list':
        symbols = list(symbols)

    # merge 一定次数, 或 maxfreq 的token pair 的出现频次 低于 阈值
    for _ in range(merge_times):

        token_pairs_w_maxfreq, maxfreq = get_maxfreq_token_pair(tokcombo_freqs) # 得到 max freq token pairs
        
        # 当 token pair occurrence freq >= min_freq 且 maxfreq > 0 时，才进行 merge 操作
        if maxfreq > 0 and maxfreq >= min_occur_freq_merge:
            # merge maxfreq token pair(s) : update vocab(symbols), tokcombo_freqs, 
            tokcombo_freqs, symbols = merge_maxfreq_token_pair(token_pairs_w_maxfreq, tokcombo_freqs, symbols, merge_mode)
        
    return symbols





# a segmenter, which tokenizes a raw sentences

def word_segment_greedy(
        word:str,
        EOW_token:str,
        symbols: t.List[str] | t.Set[str]
        ):
    '''
    input:
        word: 输入单词，用以拆分成多个 subword
        EOW_token: end-of-word token, 用以标识 word 的结尾. 在
        symbols: token set 词元集
    return:
        segmented: list of string,
            word被切分成 symbols 中包含的 subwords/tokens
        unsegmented: string
            word中未被 symbols 切分的部分。若成功切分, 则它为 空字符串
    explain:
        用贪心的方法来切割输入 word, 即用 symbols 中尽量少的 symbol 来切分 word(将 word 切割成尽量长的subwords/tokens)
    '''
    # start 是起始为止, end 是终结位置后一
    # 从start 位置开始
    #   从 end 为止开始，检查 start 到 end 是不是 symbols 中的 symbol
    #       如果不是，end 指针 往前 移 一位，重新判断
    #       如果是，记录该 symbol，同时 start 移动到 end(即终结位置后一)，end 回到末尾后一
    # 重复这个过程直到 start 等于 end
    #   start 等于 end 有两种可能：
    #       可能1: end = length 被赋值给 start. 此时即start 和 end 都处于末尾后一 位置。这意味着 word 被切割完毕
    #       可能2: end -= 1 过程中等于 start. 此时说明 word 从start位置开始，往右的每一个字符组合都不是symbols中的symbol，
    #       说明 word的start位置的字符 不存在于 symbols 中
    word += EOW_token
    length = len(word)
    start, end, segmented = 0, length, []
    while start < end:
        if word[start:end] in symbols:
            segmented.append( word[start:end] )
            start = end
            end = length
        else:
            end -= 1
    
    return segmented, word[start:]








