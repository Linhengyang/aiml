# Tokenize.py
# 本文件收录了各种 tokenize 工具。
# 至少要输入一条语句 sentence(string)，和一个词汇集 glossary。至少输出一个列表 list of tokens(list of string)

import typing as t
import re
from .TextPreprocess import preprocess_space, attach_EOW_token

# a segmenter, which tokenizes a raw sentences
def segment_word_BPE_greedy(    
        word:str,
        EOW_appnd: bool = True,
        glossary: t.Dict|None = None,
        UNK_token:str = "<unk>"
        ):
    '''
    input:
        word: str, 输入单词, 用以拆分成多个 subword. 末尾可以已经添加 EOW_token, 也可以没有添加 EOW_token(由 EOW_appnd 指明)
        若没有添加, 则 EOW_token 会被添加到末尾.

            EOW_token 由 glossary['EOW_token'] 确定, 用以标识 word 的结尾.
        
        分割的结果中, EOW_token 将以合适的方式出现: 
            若 EOW_token 不为空字符, 那么分割结果如下: 
                分割成功:       tok1, ... tokn                              EOW_token 可能被包含在 last token 内, 也可能是 独立的 last token
                分割不成功:     tok1, ... tokj, UNK_token, EOW_token        EOW_token 和 UNK_token 分别以独立 token 方式被分割
            若 EOW_token 为空字符, 那么分割结果如下:
                分割成功:       tok1, ... tokn''                            EOW_token 作为空字符, 被包含在 last token 内
                分割不成功:     tok1, ... tokj, UNK_token, ''               EOW_token 作为 独立的 空字符标识结尾

        EOW_appnd: T/F, 指明输入 word 的末尾是否已经添加 EOW_token.

        glossary: Dict, key 'EOW_token' --> end-of-word token, key 'tokens' --> 以 EOW_token 和 标准BPE流程制作的词元集.
            检查 glossary 是标准流程制作: glossary['EOW_token'] == glossary['tokens'][0]
        
            如果 glossary 不输入, 或输入 None, 那么意味着 word以 整个不分割 的方式返回

        UNK_token: str, unknown token, 用以替代 无法分割的片段

    return:
        segmented: list of string
            word被切分成 glossary['tokens'] 中包含的 subwords/tokens, 和 UNK_token(未能在 glossary['tokens'] 中匹配的部分). 以列表的方式返回
        unsegmented: string
            word中未被 glossary['tokens'] 切分的部分。若成功切分, 则它为 空字符串
    
    explain:
        用贪心的方法来切割输入 word, 即用 glossary['tokens'] 中尽量少的 symbol 来切分 word(等价于 将 word 切割成尽量长的subwords/tokens)
        以 EOW_token 作为 end-of-word token, 且以标准BPE流程制作的词元集 glossary['tokens'], EOW_token 必然以整体参与形成 token
        那么在 greedy 的算法下, 即使EOW_token 以部分参与来分割word的过程会出现, 但这种情况不会出现在最终分割结果中
    '''

    if glossary is None: # 如果 glossary 为 None, 直接将 整个word作为分割好的token返回.
        return [word], ''
    
    # 这里 EOW_token 为 null string 是可以的, 因为分割一个 word, 本来里面没有就没有空格
    EOW_token, symbols = glossary['EOW_token'], glossary['tokens']

    # start 是起始为止, end 是终结位置后一
    # 从start 位置开始
    #   从 end 为止开始, 检查 start 到 end 是不是 glossary['tokens'] 中的 symbol
    #       如果不是, end 指针 往前 移 一位, 重新判断
    #       如果是, 记录该 symbol, 同时 start 移动到 end(即终结位置后一), end 回到末尾后一
    # 重复这个过程直到 start 等于 end
    #   start 等于 end 有两种可能: 
    #       可能1: end = length 被赋值给 start. 此时即start 和 end 都处于末尾后一 位置。这意味着 word 被切割完毕
    #       可能2: end -= 1 过程中等于 start. 此时说明 word 从start位置开始, 往右的每一个字符组合都不是 glossary['tokens'] 中的symbol, 
    #       说明 word的start位置的字符 不存在于 glossary['tokens'] 中

    if not EOW_appnd: # 如果 EOW_appnd = False, 说明 word 末尾没有添加 EOW_token
        word += EOW_token
    
    length = len(word)
    start, end, segmented = 0, length, []
    while start < end:
        fragment = word[start:end]
        if fragment in symbols:
            segmented.append( fragment )
            start = end
            end = length
        else:
            end -= 1
    
    # 循环结束时 start = end. 此时有两种可能
    #   1. start = end = length, 此时 word 被切割完毕, 返回 segmented 即可. EOW_token 以合适的方式包含在了 segmented[-1] 内

    #   2. start = end < length, 此时 word 存在 不可被识别片段: 从 start位置开始. segmented 添加一个独立的 UNK 以及一个独立的 EOW
    #      以标识 不可识别片段, 和 word 的终结. 此时 返回 word 的不可识别片段作为 unsegmented part
    if start < length:
        segmented = segmented + [UNK_token, EOW_token]
    
    return segmented, word[start:]



def line_tokenize_greedy(
        sentence:str,
        glossary:t.Dict | None,
        UNK_token:str,
        flatten:bool,
        need_preprocess:bool = False,
        need_lower:bool = True, separate_puncs:str = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~', normalize_whitespace:bool = True,
        *args, **kwargs
        ):
    '''
    如果输入 glossary 是 None, 那么说明用 不切割以整体 word 作为 token 的方式来 tokenize
    贪心地用 BPE 标准流程 配合参数 EOW_token 生成的 glossary('tokens' key --> all tokens list, 'EOW_token' key --> EOW_token), 
    分割输入的 sentence
    
    根据是否 flatten, 返回 segmented_output: list of segmented tokens or list of list of segmented tokens
    
    每个 word 无法识别分割的部分, 返回 unsegmented_output: list of unsegmented string part    
    '''
    if need_preprocess:
        # 处理空白和大小写。空白：该加单空格的地方加，该改单空格的地方改，该去单空格的地方去
        # separate_puncs 确认了当作 独立token 的标点符号
        # normalize_whitespace 确认了 单空格之外的 空白字符（包括单空格之间的空字符）是否是 独立token
        sentence = preprocess_space(sentence, need_lower, separate_puncs, normalize_whitespace)

    # 如果输入的 glossary 是 None, 那么说明用 不切割以整体 word 作为 token 的方式来 tokenize
    if glossary is None:

        tokens = sentence.split(' ')

        if not flatten: # flatthen = False, 返回 2-D list of tokens
            segmented_output = [[token] for token in tokens]
        else: # flatthen = True, 返回 1-D list of tokens
            segmented_output = tokens
        
        # [] represents unsegmented part of sentence
        return segmented_output, []
    else:
        assert glossary['EOW_token'], f'EOW token of input glossary cannot be null string'

    # 统一给每个 word和 独立token 后面添加 eow_token. attach_EOW_token 本身支持 EOW_token 为 空字符, 但
    # 这里 在 切割 sentence, 所以不应该是 空字符, 不然会造成无法分辨 words 之间的自然空格 和 subword 之间的人造空格
    sentence = attach_EOW_token(sentence, glossary['EOW_token'])

    # 用单空格拆出每个 word
    words = sentence.split(' ') # 此时每个 word 是 以 EOW_token 为结尾
    segmented_output, unsegmented_output = [], []

    for word in words:
        segmented_lst, unsegmented_str = segment_word_BPE_greedy(word,
                                                                 EOW_appnd=True, # 已经统一给每个 word和 独立token 后面添加 eow_token
                                                                 glossary=glossary,
                                                                 UNK_token=UNK_token)

        if flatten:
            segmented_output.extend( segmented_lst ) # segmented_output: list of string
        else:
            segmented_output.append( segmented_lst ) # segmented_output: list of list of string

        unsegmented_output.append( unsegmented_str ) # unsegmented_output: list of string

    return segmented_output, unsegmented_output
    
