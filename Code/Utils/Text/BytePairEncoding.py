# BytePairEncoding.py
import collections
import typing as t
from .TextPreprocess import count_corpus, preprocess_space_append



# a learner, which learns from corpus to induce a vocaburary

# function byte-pair-encoding(corpus C, numbers of merges k) --> vocab V
# init V <-- all unique characters in C
# repeat k times:
#   tok_L, tok_R <-- most frequent pair of adjacent tokens in C
#   tok_new <-- tok_L + tok_R
#   V <-- V + tok_new
#   update corpus: replace all occurrence of tok_L, tok_R in C with tok_new
# return V

text = "low low lower lower lower high higher high high high"
text = preprocess_space_append(text)
# text = low_ low_ lower_ lower_ lower_ high_ higher_ higher_ higher_ high_

raw_corpus = count_corpus(text.split(" "))
symbols = set(text) | set(['<unk>'])


# corpus C 应该是个 counter, tokcombo_freq, key 是 token combo(用空格组合 token), value 是 token组成的词对应的 freq
# corpus 会更新，要区分tokens(按照 vocab V拆分tokens)
# 所以就是对 counter的 keys 作更改，即 应该合并的tokens 合并成新 token. 但是, 因为要对 每个能改的 key 都作更改，所以需要循环。而在循环中，dict的key是不能变动的
# 两个解决办法：一 每次都生成一个新的dict，把原来的dict的key-value对 都搬到新dict
# 二 counter使用 list

tokcombo_freqs = {}
for raw_word, freq in raw_corpus.items():
    tokcombo_freqs[' '.join(list(raw_word))] = freq

# tokcombo_freqs = []
# for raw_word, freq in raw_corpus.items():
#     tokcombo_freqs.append( (' '.join(list(raw_word)), freq) )


def merge_maxfreq_token_pair(maxfreq_token_pair,
                             tokcombo_freqs: t.List[t.Tuple]|t.Dict,
                             symbols:t.List|t.Set
                             ):
    '''
    input
        maxfreq_token_pair: tuple of most frequent adjacent token pair (tok_L, tok_R)
        tokcombo_freqs:
            Dict: { token_combo_by_space: word_frequency ... } / list of tuple: [ (token_combo_by_space, word_frequency) ... ]
        symbols:
            set/list
    output:
        updated tokcombo_freqs & symbols
    
    explains:
        tokcombo_freqs: tokcombo中, 最频繁出现的 连续 token pair 被合并
        symbols: 最频繁出现的 连续 token pair 被合并后, 添加入symbols
    '''
    
    if isinstance(symbols, set):
        symbols.union( ''.join(maxfreq_token_pair) )
    else:
        symbols.append( ''.join(maxfreq_token_pair) )

    if isinstance(tokcombo_freqs, dict):
        new_tokcombo_freqs = {}

        for token_combo, freq in tokcombo_freqs.items():
            # 如果该 token_combo 存在 maxfreq token pair, 合并 maxfreq_token_pair，即去掉中间的空格; 如果不存在, 那么保持原样
            new_token_combo = token_combo.replace(" ".join(maxfreq_token_pair), "".join(maxfreq_token_pair))
            new_tokcombo_freqs[new_token_combo] = freq

        return new_tokcombo_freqs, symbols
    else:
        for i, (token_combo, freq) in enumerate(tokcombo_freqs):
            maxfreq_token_pair_combo_by_space = " ".join(maxfreq_token_pair)

            if maxfreq_token_pair_combo_by_space in token_combo:
                # 如果该 token_combo 存在 maxfreq token pair, 合并 maxfreq_token_pair，即去掉中间的空格
                new_token_combo = token_combo.replace(maxfreq_token_pair_combo_by_space, "".join(maxfreq_token_pair))
                tokcombo_freqs[i] = (new_token_combo, freq)

        return tokcombo_freqs, symbols







# a segmenter, which tokenizes a raw sentences