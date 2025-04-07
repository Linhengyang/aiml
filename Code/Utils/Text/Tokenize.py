# Tokenize.py
# 本文件收录了各种 tokenize 工具。
# 至少要输入一条语句 sentence(string)，和一个词汇集 symbols(set|list)。至少输出一个列表 list of tokens(list of string)


import typing as t
import re
from .TextPreprocess import preprocess_space, attach_EOW_token
from .BytePairEncoding import segment_word_BPE_greedy





def line_tokenize_simple(
        sentence:str,
        *args, **kwargs
        ) -> t.List[str]:
    
    return sentence.split(' ')



def line_tokenize_greedy(
        sentence:str,
        symbols:t.List[str] | t.Set[str],
        EOW_token:str,
        UNK_token:str = "<unk>",
        need_lower:bool = True,
        flatten:bool = True,
        separate_puncs:str = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
        normalize_whitespace:bool = True,
        *args, **kwargs
        ):

    # 处理空白和大小写。空白：该加单空格的地方加，该改单空格的地方改，该去单空格的地方去
    # separate_puncs 确认了当作 独立token 的标点符号
    # normalize_whitespace 确认了 单空格之外的 空白字符（包括单空格之间的空字符）是否是 独立token
    sentence = preprocess_space(sentence, need_lower, separate_puncs, normalize_whitespace)

    # 给每个 word和 独立token 后面添加 eow_token
    sentence = attach_EOW_token(sentence, EOW_token)

    # 用单空格分割
    words = sentence.split(' ') # 每个 word 是 以 EOW_token 为结尾
    segmented_output, unsegmented_output = [], []

    for word in words:
        segmented_lst, unsegmented_str = segment_word_BPE_greedy(word, symbols, UNK_token, EOW_token)

        if flatten:
            segmented_output = segmented_output + segmented_lst # segmented_output: list of string
        else:
            segmented_output.append( segmented_lst ) # segmented_output: list of list of string

        unsegmented_output.append( unsegmented_str ) # unsegmented_output: list of string

    return segmented_output, unsegmented_output
    
