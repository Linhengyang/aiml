# Tokenize.py
# 本文件收录了各种 tokenize 工具。
# 至少要输入一条语句 sentence(string)，和一个词汇集 symbols(set|list)。至少输出一个列表 list of tokens(list of string)

import typing as t
import re
from .TextPreprocess import preprocess_space, attach_EOW_token
from .BytePairEncoding import segment_word_BPE_greedy





def line_tokenize_greedy(
        sentence:str,
        symbols:t.List[str] | t.Set[str] | None,
        EOW_token:str,
        UNK_token:str,
        flatten:bool,
        need_preprocess:bool = False,
        need_lower:bool = True, separate_puncs:str = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~', normalize_whitespace:bool = True,
        *args, **kwargs
        ):
    '''
    贪心方式, 用 BPE 标准流程 配合参数 EOW_token 生成的 symbols, 分割输入的 sentence
        根据是否 flatten, 返回 segmented_output: list of segmented tokens or list of list of segmented tokens
        每个 word 无法识别分割的部分, 返回 unsegmented_output: list of unsegmented string part
        
    如果输入 symbols 是 None 或者 空 []/set(), 那么说明用 不切割以整体 word 作为 token 的方式来 tokenize
    '''
    if need_preprocess:
        # 处理空白和大小写。空白：该加单空格的地方加，该改单空格的地方改，该去单空格的地方去
        # separate_puncs 确认了当作 独立token 的标点符号
        # normalize_whitespace 确认了 单空格之外的 空白字符（包括单空格之间的空字符）是否是 独立token
        sentence = preprocess_space(sentence, need_lower, separate_puncs, normalize_whitespace)

    # 如果输入的 symbols 是 空, 或者是 None, 那么说明用 不切割以整体 word 作为 token 的方式来 tokenize
    if symbols is None or len(symbols) == 0:

        tokens = sentence.split(' ')

        if not flatten: # flatthen = False, 返回 2-D list of tokens
            segmented_output = [[tokens] for token in tokens]
        else: # flatthen = True, 返回 1-D list of tokens
            segmented_output = tokens
        
        # [] represents unsegmented part of sentence
        return segmented_output, []

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
    
