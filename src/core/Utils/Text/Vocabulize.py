# Vocabulize.py
import typing as t
from .text_preprocess import count_corpus
from .string_segment import sentence_segment_greedy, Glossary
import json
import os
import copy


class Vocab:
    '''
    args:
        corpus: 建立 Vocab 所需要的基本语料库. 默认 null string
        glossary: 切分 corpus 到 token list 所用的词汇集. 如果是 None, 则用 单空格 切分 corpus. 默认 None
        need_lower: 默认 True
        reserved_tokens(optional): user-reserved special tokens. unk_token/EOW_token of glossary are added in default.
        min_freq(optional): threshold to map rare tokens to unk_token. frequency strictly under it will be map to unk_token
    
    如果不输入参数初始化, 即全部采用默认输入, 得到的是 空 vocab, 可以通过 load vocab fpath 来加载 vocab

    returns:
        A mapper whose main function is to map tokens(words, chars) to indices, or vice versa.
    
    explains:
        定义一个vocab类, 它提供将token映射到integer ID的功能

        本 Vocab 得到的词汇量, 是所使用的 glossary 的高频子集. 低频 token <==> unk_token
        故对于某语言的语料数据 data 来说,
            data --tokenize by glossary+unk--mapping by vocab--> numeric data, 等价于
            data --tokenize and mapping by vocab----> numeric data
        
        vocab 类和 glossary 的区别:
        vocab 是语料 corpus 基于 glossary 的再加工.
            glossary作为一个dict, 只记录了BPE流程制作的all tokens, 以及所使用的 EOW_token. 没有统计信息, 不包含 unk_token

            而vocab基于 glossary, 对语料 corpus 再加工, 基于 glossary tokenize corpus 后得到的 tokens 的统计频次, 提供 map token to ID 的功能.
            vocab.unk 就是 unk_token 的 ID, vocab.eow 是所基于的 glossary 的 EOW_token. 如果 它是 '', 说明原 glossary 是 None
    '''
    def __init__(self,
                 corpus:str='',
                 glossary:Glossary|None=None,
                 need_lower:bool=True,
                 reserved_tokens:t.List[str]=[],
                 unk_token='<unk>',
                 separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
                 min_freq=1,
                 ):

        # 将 corpus 作为 一条string 输入, normalize 空白格式,独立化每个 punc
        # sentence_segment_greedy 要求 glossary 要么为 None, 要么 EOW_token 不为空
        tokens = sentence_segment_greedy(corpus, glossary, unk_token,
                                         flatten = True,
                                         need_preprocess = True,
                                         need_lower = need_lower,
                                         separate_puncs = separate_puncs,
                                         normalize_whitespace = True)[0]
        # 使用 给定的 glossary 切割 corpus, 得到 tokens: list of tokens

        # 如果 glossary 不为 None, 那么:
        #   得到的 tokens中, 原始 word 的结尾token, 会用 glossary['EOW_token'] 标识
        #   得到的 tokens 一定都是存在于 glossary['tokens'] + unk_token 里.

        # 所以显然, 本vocab构建的 token 集合, 是 glossary['tokens'] + unk_token 的子集

        counter = count_corpus(tokens) # 计算各个 token 出现频次
        counter.pop('', None) # 去除可能存在的 '' 空字符串token

        EOW_token = glossary['EOW_token'] if glossary else '' # 记录 EOW_token. 
        # 如果 glossary 是 None, 那么 EOW_token 设为 空字符串 ''
        # 如果 glossary 不为 None, 那么 EOW_token 为 非空字符串

        # ----> 可以通过 本 vocab 的 EOW_token 是否为 '' 判断 所使用的 glossary 是否是 None

        del tokens, glossary # 节省内存

        # 将(token, frequency)组成的pair, 按照frequency从多到少排序记录在一个list里
        # sorted()函数, 输入一个iterable对象, 按照key参数排序, 默认从小到大排序, reverse=True则为从大到小排序）
        # 返回一个list, 元素是(key, value)对
        self._token_freqs = sorted(counter.items(), key=lambda x:x[1], reverse=True)

        # 建立从token到数字的双射

        ## 更新 reserved_tokens: 
        # 添加 unk 到 index0 位置, 添加 eow 到 index1 位置
        self._reserved_tokens = [unk_token, EOW_token] + reserved_tokens

        # 记录 token 而不是 unk_token 的 最小出现 频次
        self._min_freq = min_freq


        # 构建 vocab 的两个 mapping:
        self.__build_mapping()
    
    
    def __build_mapping(self):
        # 构建 vocab 的两个 mapping:
        #   _idx_to_token, a list mapping idx to token
        #   _token_to_idx, a dict mapping token to idx

        ## 初始化token列表, 以便用index取出token, 即从 数字映射到token
        self._idx_to_token = copy.deepcopy( self._reserved_tokens )
        ## 初始化token-idx字典, 以便用取value的方式取idx, 即从token映射到数字
        self._token_to_idx = {token: idx for idx, token in enumerate(self._idx_to_token)}

        ## 遍历_token_freqs中的所有token, 按序添加到self.idx_to_token, 并添加token-idx的kv pair到token_to_idx
        for token, freq in self._token_freqs:
            if freq < self._min_freq: # 如果该token出现的频次小于阈值, 则视该token为<unk>, 即已经建立好双射了. 后续的也不需要再建立映射, 跳出循环
                break
            else:
                if token not in self._reserved_tokens: # reserved_tokens 内存在的 token 已经在idx_to_token和token_to_idx中建立好双射关系了
                    self._idx_to_token.append(token) # 添加该token到self.idx_to_token
                    self._token_to_idx[token] = len(self._idx_to_token) - 1 # 添加token-idx的kv pair到token_to_idx


    def __len__(self):
        '''提供len()函数接口, 返回该vocab中总共有多少个distinct token'''
        return len(self._idx_to_token)
    
    
    # 对给定的token（或者是tokens的hierarchy组合体）, 返回idx（或者是indices对应的hierarchy组合体）
    def __getitem__(self, tokens):
        '''提供文字to数字映射, 对给定的token, 返回其对应的idx. 如果输入有hierarchy结构, 则以相同的结构返回'''
        if not isinstance(tokens, (list, tuple)): # 当tokens不是容器类型时（即tokens是string时）
            # 字典的get方法, 取出tokens对应的value, 返回. 如果取不到, 即是未知元, 返回self.unk, 即0
            return self._token_to_idx.get(tokens, self.unk)
        # 当tokens是容器类型时, 取出内部的元素, 递归式地调用本函数. 这样本函数的返回值和输入值token具有相同的hierarchy结构
        return [self.__getitem__(token) for token in tokens]
    

    # 对给定的idx（或者是indices的hierarchy组合体）, 返回token（或者是tokens对应的hierarchy组合体）
    def to_tokens(self, indices):
        '''提供数字to文字映射, 对给定的idx, 返回其对应的token. 如果输入有hierarchy结构, 则以相同的结构返回'''
        if not isinstance(indices, (list, tuple)):
            try:
                return self._idx_to_token[indices]
            except IndexError:
                return self._idx_to_token[ self.unk ] # 如果出现indexerror, 返回 UNK_token
        
        return [self.to_tokens(index) for index in indices]
    

    @property
    def unk(self) -> int:
        '''未知元 unk_token(默认为<unk>) 的idx为0'''
        return 0
    

    @property
    def eow(self) -> str:
        '''制作本 vocab 时基于的 glossary 的 end-of-word token 的idx为0.
        如果 eow_token 是null string, 说明glossary是None, 本 vocab 完全根据输入 corpus 的单词 word 来构建'''
        return 1

    
    @property
    def token_freqs(self) -> t.List[t.Tuple]:
        return self._token_freqs
    

    @property
    def tokens(self) -> t.List[str]:
        return self._idx_to_token
    

    @property
    def reserved_tokens(self) -> t.List[str]:
        return self._reserved_tokens
    

    def save(self, fpath, format='json'):
        # vocab 只需保存 _token_freqs list of (token, freq), _reserved_tokens list of (unk_token, eow_token,...), _min_freq
        # 就完整保存了 整个 vocab 的信息
        if format == 'json':
            # 以 json 格式保存 一个 dict 对象
            with open(fpath, 'w') as f:
                json.dump(
                    {'_token_freqs':self._token_freqs,
                     '_reserved_tokens':self._reserved_tokens,
                     '_min_freq':self._min_freq
                     }, f)
        else:
            raise NotImplementedError(
                f'format {format} not implemented')
    
    
    def load(self, fpath, format='json'):
        # vocab 只需保存 _token_freqs list of (token, freq), _reserved_tokens list of (unk_token, eow_token,...), _min_freq
        # 就完整保存了 整个 vocab 的信息
        if format == 'json':
            # 分别 load 到 self.idx_to_token 和 self.token_to_idx
            with open(fpath, 'r') as f:
                vocab_info = json.load(f)
        else:
            raise NotImplementedError(
                f'format {format} not implemented')
        
        self._token_freqs = vocab_info['_token_freqs']
        self._reserved_tokens = vocab_info['_reserved_tokens']
        self._min_freq = vocab_info['_min_freq']

        # 构建 vocab 的两个 mapping:
        self.__build_mapping()