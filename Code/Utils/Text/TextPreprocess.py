import collections
import random
import math
import typing as t



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



def preprocess_space(text, need_lower=True, separate_puncs=',.!?'):
    '''
    inputs: text, need_lower(optional), separate_puncs(optional)
        text: a str object
        need_lower: Bool, default as True. If true, then input str will be lowered
        separate_puncs: punctuations that shall be seen as independent tokens, such as ,.!?

    returns: A str obejct
        whose spaces are normal single space ' ', and single space is inserted before every independent token

    explains:
        preprocess spaces inside a str obeject
    '''
    text = text.replace('\u202f', ' ').replace('\xa0', ' ') #替换不间断空格为单空格
    if need_lower:
        text = text.lower()
    # 在文字和[,.!?]之间插入空格
    ## 判断 当前字符是否是separate_puncs，且前一个字符不是空格
    def no_space(char, prev_char):
        return char in set(separate_puncs) and prev_char != " "
    ## 从第二个字符开始遍历。如果它是,.!?且前一个字符不是空格，则将它变成 " "+标点
    out = [ " " + char if i > 0 and no_space(char, text[i-1]) else char for i, char in enumerate(text)]
    return "".join(out)



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