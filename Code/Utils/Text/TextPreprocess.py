import collections
import random
import math
import typing as t
import re



def count_corpus(sentences: t.List[list]|t.List[str]) -> t.Dict:
    '''
    inputs: sentence/s of tokens
        sentences can be 1D list or 2D list with basic elements as tokens

    returns: A dictionary, denoted as D
        keys are tokens(words, chars) and values are frequencies

    explains:
        Count token frequencies 
    '''
    # flatten the 2D list
    if len(sentences) == 0 or isinstance(sentences[0], list):
        sentences = [token for line in sentences for token in line]

    return collections.Counter(sentences)



def preprocess_space(
        text,
        need_lower=True,
        separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
        normalize_whitespace=True
        ) -> str:
    '''
    inputs:
        text
        need_lower: Bool, default as True. If true, then input str will be lowered
        separate_puncs: punctuations that shall be seen as independent tokens, such as ,.!?
        normalize_whitespace: False -> 制表符换行符等其他种类空白字符保留，连续空格保留；True -> 以上全部转换为 单个单空格

    returns: A str obejct
        whose spaces are normal single space ' ', and single space is inserted before every independent token

    explains:
        preprocess spaces inside a str obeject
        参数 separate_puncs 确认了 作为独立token的标点符号
        参数 normalize_whitespace 确认了如何处理 单空格之外的空白字符。
        当 它为 True 时，所有 空白字符和连续单空格都被处理为 单个单空格，所以只有 word 和 punc 被 append tok
        当 它为 False时，诸如制表符和换行符之类的空白字符，以及连续单空格都被保留，所以 空白字符和空字符 也会被 append tok
    '''
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').strip() #替换不间断空格为单空格, 并trim首尾空格

    if need_lower:
        text = text.lower()
    
    # 在文字和[,.!?]之间插入空格
    ## 判断 当前字符是否是separate_puncs，且前一个字符不是空格
    def no_space(char, prev_char):
        return char in set(separate_puncs) and prev_char != " "
    ## 从第二个字符开始遍历。如果它是separate_puncs且前一个字符不是空格，则将它变成 " "+标点
    out = [ " " + char if i > 0 and no_space(char, text[i-1]) else char for i, char in enumerate(text)]
    out_str = "".join(out)

    # 如果 normalize_whitespace = True, 那么把 所有其他种类的空白字符(\t \n)以及多个连续的单空格 转换为 单个单空格 
    if normalize_whitespace:
        out_str = re.sub(r'\s+', ' ', out_str)
    
    return out_str



def subsample(sentences:t.List[t.List[str]], vocab, thr=1e-4):
    '''
    sentences 是2D list of lists of text-tokens.
    vocab是sentences的词汇字典
    
    subsample是降采样(下采样), 即对高频(频率高于thr)的token, 以 sqrt( thr /freq_rate(token) )的概率保留
    <unk>未知字符代表的是所有「超低频」字符, 所以不应该带在降采样之列. 在subsample过程中会将<unk>以0概率保留(即去除)

    return:
        subsampled sentences(word tokens) 
        counter(count every token's frequency except <unk> before subsampling)
    '''
    # exclude <unk>
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    # count all tokens
    counter = count_corpus(sentences)
    num_all_tokens = sum(counter.values())

    def keep(token):
        return random.uniform(0, 1) < math.sqrt(thr/ counter[token] * num_all_tokens)

    return [[token for token in line if keep(token)] for line in sentences], counter



def preprocess_appdtokn_b4_space(
        text,
        append_tok
        ) -> str:
    '''
    inputs:
        text
        append_tok: 该 append_tok 将被插入到每个 单空格之前

    returns:
        whose spaces are normal single space ' ', and every space has append_punc append before it

    explains:
        因为 subword 会拆分整个word, append_tok 帮助区分subword之间和word之间的分割。该 append_tok 将被插入到每个 单空格之前
    '''
    text = text.strip() + " "

    words_and_puncs = text.split(" ") # 末尾包含 一个空字符串 ""
    _SPACE = append_tok+" "
    
    return _SPACE.join(words_and_puncs).strip() # 剪去 最后面的 空格



def text_atomize(text, tail_token:str = '', reserved_combos:t.List[str]=[], uniq=False) -> t.List[str]:
    '''
    按顺序分割字符串到不可分割字符（保留reserved_combos里的组合不拆分，其他拆分成单个字符；tail_token和它前一个字符不被分割）
    tail_token 不能出现在 reserved_combos 的任意一个组合中
    返回list
    '''
    rsvd_tokn_pattern = "(" + "|".join(reserved_combos) + ")" # (<unk>|...|<eos>)

    if re.search(tail_token, rsvd_tokn_pattern): # 如果在 rsvd_tokn_pattern 中检测到 tail_token
        raise ValueError(f'tail_token {tail_token} exists in reserved_combos {reserved_combos}. must get rid of tail_token')
    
    # 如果 reserved_combos 为空, 那么就是不需要保留组合，直接返回 list(text)
    if not reserved_combos:
        return list(text)
    
    pattern = re.compile( "(" + "|".join(reserved_combos) + ")" + "|" + "|(.)" ) # (<unk>|...|</w>)|(.)  保留字符匹配group1，其他所有单个字符匹配group2
    result = []
    for match in re.finditer(pattern, text):
        result.append( match.group(1) if match.group(1) else match.group(2) ) # 如果 match 匹配的是group1, 保留词；如果 match 匹配的是group2, 任意其他字符

    if uniq:
        result = list( set(result) )
    
    return result