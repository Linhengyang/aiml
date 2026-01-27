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
            glossary作为一个dict, 只记录了BPE流程制作的所有tokens, 以及所使用的 EOW_token. 没有统计信息, 不包含 unk_token
            而vocab基于 glossary, 对语料 corpus 再加工, 基于 glossary tokenize corpus 后得到的 tokens 的统计频次, 提供 map token to ID 的功能.
            vocab.unk 就是 unk_token 的 ID, vocab.eow 是所基于的 glossary 的 EOW_token. 如果 它是 '', 说明原 glossary 是 None
    '''
    def __init__(self,
                 corpus:str='',
                 glossary:Glossary|None=None,
                 need_lower:bool=True,
                 reserved_subwords:t.List[str]=[],
                 unk='<unk>',
                 separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
                 min_freq=1,
                 ):

        # 将 corpus 作为 一条string 输入, normalize 空白格式,独立化每个 punc
        # sentence_segment_greedy 要求 glossary 要么为 None, 要么 EOW_token 不为空字符串
        tokens = sentence_segment_greedy(corpus, glossary, unk,
                                         flatten = True,
                                         need_preprocess = True,
                                         need_lower = need_lower,
                                         separate_puncs = separate_puncs,
                                         normalize_whitespace = True)[0]
        # 使用 给定的 glossary 切割 corpus, 得到 tokens: list of subwords

        # 如果 glossary 不为 None, 那么:
        #   得到的 tokens中, 原始 word 的结尾token, 会用 glossary['EOW_token'] 标识
        #   得到的 tokens 一定都是存在于 glossary['tokens'] + unk_token 里.
        # 所以显然, 本vocab构建的 token 集合, 是 glossary['tokens'] + unk_token 的子集
        counter = count_corpus(tokens) # 计算各个 token 出现频次
        counter.pop('', None) # 去除可能存在的 '' 空字符串token
        eow = glossary['EOW_token'] if glossary else '' # 记录 EOW_token 到 _reserved_subwords 的 第二个位置
        # 如果 glossary 是 None, 那么 eow 设为 空字符串 ''
        # 如果 glossary 不为 None, 那么 eow 为 非空字符串
        # ----> 可以通过 本 vocab 的 eow subword 是否为 '' 判断 所使用的 glossary 是否是 None
        del tokens, glossary # 节省内存
        # 将(token, frequency)组成的pair, 按照frequency从多到少排序记录在一个list里
        # sorted()函数, 输入一个iterable对象, 按照key参数排序, 默认从小到大排序, reverse=True则为从大到小排序）
        # 返回一个list, 元素是(key, value)对
        self._subword_freqs = sorted(counter.items(), key=lambda x:x[1], reverse=True)
        # 建立从 subword 到数字的双射
        ## 更新 reserved_subwords: 
        # 添加 unk 到 index0 位置, 添加 eow 到 index1 位置
        self._reserved_subwords = [unk, eow] + reserved_subwords
        # 记录 token 而不是 unk_token 的 最小出现 频次
        self._min_freq = min_freq
        # 构建 vocab 的两个 mapping:
        self.__build_mapping()
    
    def __build_mapping(self):
        # 构建 vocab 的两个 mapping:
        #   _idx_to_subword, a list mapping idx to subword
        #   _subword_to_idx, a dict mapping subword to idx

        ## 初始化 subword 列表, 以便用index取出token, 即从 数字映射到 subword
        self._idx_to_subword = copy.deepcopy( self._reserved_subwords )
        ## 初始化token-idx字典, 以便用取value的方式取idx, 即从token映射到数字
        self._subword_to_idx = {subword: idx for idx, subword in enumerate(self._idx_to_subword)}

        ## 遍历 _subword_freqs 中的所有token, 按序添加到self._idx_to_subword, 并添加subword-idx的kv pair到subword_to_idx
        for subword, freq in self._subword_freqs:
            if freq < self._min_freq: # 如果频次小于阈值, 则视该为<unk>, 即已经建立好双射了. 后续的也不需要再建立映射, 跳出循环
                break
            else:
                if subword not in self._reserved_subwords: # _reserved_subwords 内存在的 token 已经在 idx_to_subword 和 subword_to_idx 中建立好双射关系了
                    self._idx_to_subword.append(subword) # 添加该 subword 到self.idx_to_subword
                    self._subword_to_idx[subword] = len(self._idx_to_subword) - 1 # 添加 subword-idx 的kv pair到 subword_to_idx

    def __len__(self):
        '''提供len()函数接口, 返回该vocab中总共有多少个distinct token'''
        return len(self._idx_to_subword)
    
    # 对给定的 subwords（或者是 subwords 的hierarchy组合体）, 返回idx（或者是indices对应的hierarchy组合体）
    def __getitem__(self, subwords):
        '''提供文字to数字映射, 对给定的token, 返回其对应的idx. 如果输入有hierarchy结构, 则以相同的结构返回'''
        if not isinstance(subwords, (list, tuple)): # 当tokens不是容器类型时（即tokens是string时）
            # 字典的get方法, 取出tokens对应的value, 返回. 如果取不到, 即是未知元, 返回self.unk, 即0
            return self._subword_to_idx.get(subwords, self.unk)
        # 当tokens是容器类型时, 取出内部的元素, 递归式地调用本函数. 这样本函数的返回值和输入值token具有相同的hierarchy结构
        return [self.__getitem__(subword) for subword in subwords]

    # 对给定的idx（或者是indices的hierarchy组合体）, 返回 subword（或者是 subword 对应的hierarchy组合体）
    def to_subwords(self, indices):
        '''提供数字to文字映射, 对给定的idx, 返回其对应的token. 如果输入有hierarchy结构, 则以相同的结构返回'''
        if not isinstance(indices, (list, tuple)):
            try:
                return self._idx_to_subword[indices]
            except IndexError:
                return self._idx_to_subword[ self.unk ] # 如果出现indexerror, 返回 UNK_token
        
        return [self.to_subwords(index) for index in indices]
    
    @property
    def unk(self) -> int:
        '''未知元 unk_token(默认为<unk>) 的idx为0'''
        return 0
    
    @property
    def eow(self) -> str:
        '''
        制作本 vocab 时基于的 glossary 的 end-of-word token 的idx为 1
        如果 eow subword 是null string, 说明glossary是None, 本 vocab 完全根据输入 corpus 的单词 word 来构建
        '''
        return 1
    
    @property
    def subwords(self) -> t.List[str]:
        return self._idx_to_subword
    
    @property
    def reserved_subwords(self) -> t.List[str]:
        return self._reserved_subwords

    @property
    def subword_freqs(self) -> t.List[t.Tuple]:
        return self._subword_freqs

    def save(self, fpath, format='json'):
        # vocab 只需保存 _subword_freqs list of (subword, freq), _reserved_subwords list of (unk, eow,...), _min_freq
        # 就完整保存了 整个 vocab 的信息
        if format == 'json':
            with open(fpath, 'w') as f:
                json.dump(
                    {'_subword_freqs':self._subword_freqs,
                     '_reserved_subwords':self._reserved_subwords,
                     '_min_freq':self._min_freq
                     }, f)
        else:
            raise NotImplementedError(
                f'format {format} not implemented')
    
    def load(self, fpath, format='json'):
        if format == 'json':
            with open(fpath, 'r') as f:
                vocab_info = json.load(f)
        else:
            raise NotImplementedError(
                f'format {format} not implemented')
        self._subword_freqs = vocab_info['_subword_freqs']
        self._reserved_subwords = vocab_info['_reserved_subwords']
        self._min_freq = vocab_info['_min_freq']
        # 构建 vocab 的两个 mapping:
        self.__build_mapping()

    def to_glossary(self) -> Glossary|None:
        if self.to_subwords(1): # 当 eow 不是 null string
            return Glossary(tokens=self._idx_to_subword, EOW_token=self._idx_to_subword[1])
        else:
            return None