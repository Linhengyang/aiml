# Tokenize.py
# 本文件收录了各种 tokenize 工具。
# 至少要输入一条语句 sentence(string)，和一个词汇集 symbols(set|list)。至少输出一个列表 list of tokens(list of string)


import typing as t
import re






def line_tokenize_simple(
        sentence:str,
        symbols: t.List[str] | t.Set[str] | None = None
        ) -> t.List[str]:
    
    return sentence.split(' ')



def line_tokenize_