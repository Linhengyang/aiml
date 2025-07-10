# Tokenizer.py
# new version of Tokenizer inspired by GPT tokenizer(tiktokenizer)

# 新版本的 Tokenizer 应该满足：
# 1. 一站式满足 encode（original string --> indices）, decode (indices --> original string），且保证是 round-trips
# 2. 可序列化
# 3. 可扩展
# 4. 高效
# 5. 行为对齐：word with its preceding space are part of the same token，即 单空格作为原始字符串参与tokenize
#             在 Glossary 中实现的经典 BPE 方法，实际蕴含了一个 “ 预分词 ” 步骤，即整个文本流（添加EOW后），用单空格切分，然后基于切分的结果作 BPE 进一步切分
#             这里的 用单空格切分，就是 预分词。预分词其实否定了“LLM学习单空格”：即单空格不是需要学习的token（甚至不是需要学习的token的一部分）。
#             它依赖于语言的空格约定。并且对 python 这样的代码文本也不友好（因为python代码里空格蕴含语义信息）。

#             一, 预分词可以改进，比如英文文献可以使用 GPT2_TOKENIZER_REGEX 来 预分词。
#             二，预分词可以彻底摒弃：现代的 Tokenizer 比如 SentencePiece，不依赖 pre-tokenize 或 语言的空格约定，它将整个文本流视作一个序列，并学习子词
#             它将空格显示地包含在 token 中（通常作为前缀），明确地标记了单词的开始，从而简化了逆向转换过程，也能确保信息不丢失。特别适合多语言环境。
#             在学习过程中，首先把单空格转化为一个特殊、可见的符号（U+2581），然后它与其他字符一起参与BPE学习。由于自然语言中的空格真实频率，大量自带前导空格的
#             token 会出现在词表中，代表了新word的开始。 好处是1：文本序列（不需要eow） <--> token 序列相互 转换无损，2: 不依赖语言的空格约定，对东亚语言友好
# 6. 训练过程: 字节级 byte-level BPE 以解决 out-of-vocabulary 问题。BBPE 把所有 原始字符串 转换成 字节序列，然后从 UTF-8 字节序列（而不是 unicoode字符序列）
#             初始开始。意味着 初始 词汇表是 256 个 可能的字节，然后 迭代地合并出现最频繁的字节对。
#             Byte-BPE 和 Char-BPE 的 merge 过程是相同的。
#             Char-BPE 的缺点在于 OOV 问题 和 无损 round-trips 无法同时保证。Char-BPE 的最原子字符集是 train corpus 的原子字符集，但是若在推理tokenize时
#             遇到了不在 train corpus 原子字符集 中的 character，就会把它以及它后面chunk 部分映射到 UNK token。然而 UNK token 导致了 encode-decode 不是
#             round-trips（不是无损编解码）。
#             Byte-BPE 不会有 类似的问题，因为无论是否 出现在 train corpus 中的字符，都是 unicode 字符，由utf-8字节组成，其 encode-decode 是 round-trips。

#             当然 Byte-BPE 会引入独属于它自己的问题，即跨字符分割可能出错（英文是单字节的，不存在跨字符分割；欧洲和中日韩文是2-3字节的，可能会在encode时，
#             把属于同一个字符的字节，分割到了不同token里。这样会使得 流式decode 时出现 UnicodeDecodeError：比如 某个 token(此时它是字节序列)要么多了个字节，
#             要么少了字节。所以 Byte-BPE 的 解码器要特别编写。
#             Char-BPE 的 流式解码器可以即时输出，比如拿到一个 token 就马上打印（因为此时它是字符序列）
#             但是 Byte-BPE 的流式解码器需要一个 queue 来 buffer 输出，生成的 tokens 不断进入 queue。流式解码器从 queue 的头部开始，依次尝试解码 1 2 3 4字节
#             以生成字符，若成功生成字符，则消耗对应字节；若生成失败，则停止输出等待下一个token进入。全部生成结束后，对 queue 中剩余字节作最后尝试。
#             不过 非流式解码器（生成全部token之后，再一次性输出）不存在这个问题。比如 经典 transformer 中的 beam search，需要对 生成 sequence 的长度奖惩，
#             所以需要生成 num_steps 步后，再一起输出。这种情况下跨字符tokenize的问题会比较轻。 

# 7. 推理过程：这个还是 “最长贪婪匹配”。但实现过程不一样：StringSegment.py 里写的是从末尾开始查找尽量长的token，而现代 Tokenizer 一般实现一个 树结构，以便
#             从头开始高效查找词汇表中的最长匹配（无需遍历整个词汇表）


# 一个合适的 tokenizer：合适的压缩率，使得 string tokenized 之后的 token 数量少，这样 attention 机制能尽量抓住序列信息（attention 对序列长度的消耗是L^2）
# 尽量少的 token 数量，要求 vocab_size 尽量大。但过大的 vocab_size 将使得 next token prediction 的softmax 机制不准确。
# meta-class
from abc import ABC
import typing as t


class Tokenizer(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def encode(self, string: str) -> t.List[int]:
        raise NotImplementedError
    
    def decode(self, indices: t.List[int]) -> str:
        raise NotImplementedError



class CharacterTokenizer(Tokenizer):
    '''
    A tokenizer mapping between unicode characters and its unicode number as tokens
    encode: char --> integer between 0 and 0x10FFFF
    decode: integer between 0 and 0x10FFFF --> char
    '''
    def encode(self, string:str) -> t.List[int]:
        return list( map(ord, string) )
    
    def decode(self, indices: t.List[int]) -> str:
        # filter valid unicode index
        indices = [ i  for i in indices if 0<= i <= 0x10FFFF ]
        return "".join(map(chr, indices))
    


class ByteTokenizer(Tokenizer):
    '''
    A tokenizer splits string into bytes, and use integers between 0 and 255 as tokens
    encode: string --> integer(between 0 and 255) sequence
    decode: integer(between 0 and 255) sequence --utf-8--> string with possible replacement
    '''
    def encode(self, string: str) -> t.List[int]:
        string_bytes = string.encode("utf-8") # 返回 utf-8 规范编码的 字节byte 序列. 所以返回的 string_bytes 是一个序列
        # 可以求 len, 返回 字节数量; 可以索引 index，返回各个 字节的整数值(0-255)
        # 英文 1 字节，欧洲字符 2字节，中日韩字符 3字节，罕见字符 4字节
        indices = list( map(int, string_bytes) ) # list of integers btw 0-255
        return indices
    
    def decode(self, indices: t.List[int]) -> str:
        # filter valid unicode index
        try:
            string_bytes = bytes(indices) # bytes 其中一种使用方式: 输入 list of integers, 要求每个 integer 0-255
            return string_bytes.decode('utf-8', errors='replace')
        except ValueError:
            print(f'input indices {indices} has wrong values. must be 0-255')






import regex as re
import os
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from ..common.seq_operation import check_monotonic


# deprecated split-pattern for GPT2. use GPT4 version
GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


GPT4_TOKENIZER_REGEX = \
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


ENDOFTEXT = '<|endoftext|>'
FIM_PREFIX = '<|fim_prefix|>'
FIM_MIDDLE = '<|fim_middle|>'
FIM_SUFFIX = '<|fim_suffix|>'
ENDOFPROMPT = '<|endofprompt|>'




def get_pair_counts(tokens:t.List[int], p_counts:t.Dict[tuple[int, int], int]|None = None):
    '''
    p_counts 如果是 None: init a p_counts and return
        给定 tokens list, 计算它的 pair-tokens counts, 返回一个 pair counts dict
    p_counts 如果不是 None: in-place update p_counts
        给定 tokens list, 计算它的 pair-tokens counts, 更新 到 输入的 p_counts 里
    '''
    if p_counts is None:
        p_counts = {}
    if len(tokens) == 1: # 如果只剩下 一个 token, 那么就无需 count pair / 更新 p_counts 
        return p_counts
    
    for pair in zip(tokens, tokens[1:]):
        p_counts[pair] = p_counts.get(pair, 0) + 1

    return p_counts



def merge_pair(tokens:t.List[int], pair:tuple[int, int], new_token):
    new_tokens = [] # 返回一个 new_tokens, 而不是对 tokens 作 in-place 更新
    i = 0
    if len(tokens) == 1: # 如果只剩下 一个 token, 那么就没有 merge pair 的必要
        return tokens
    
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_tokens.append( new_token )
            i += 2
        else:
            new_tokens.append( tokens[i] )
            i += 1
    return new_tokens






# 缓存这个函数, 因为预计会用完全相同的输入，调用很多次这个函数。缓存函数调用结果，下次遇到相同输入时，避免计算直接返回缓存结果
@functools.lru_cache(maxsize=128)
def _special_token_regex(tokens: frozenset[str]) -> "re.Pattern[str]":
    inner = "|".join(re.escape(token) for token in tokens)
    # 返回诸如 "(<pad>|<eos>|<bos>)" 之类的 special tokens 或运算符 拼接的 compiled regex
    # compiled regex 可以直接调用 .search(text) 方法
    return re.compile(f"({inner})")



def raise_run_out_corpus_error(num_occured_merges:int, num_specials:int) -> t.NoReturn:
    '''
    如果经过 tokens 累积更新 p_counts, p_counts 仍然是 {}, 说明 corpus 已经全部 merge 到一起, 无可 merge。
    不提前结束 merge 循环, 而是报错, 提示 更换参数 explicit_n_vocab 或 语料 corpus。
    原因是参数 explicit_n_vocab 语意为「明确的 vocab_size」，
    所以不应引入不确定性：当 explicit_n_vocab 和 corpus 冲突时，raise error而不是根据 corpus 确定 explicit_n_vocab。
    '''
    raise RuntimeError(f'run out of corpus(all merged together) after {num_occured_merges} merges.\n'
                       f'the maximum valid `explicit_n_vocab` for this corpus is {256+num_occured_merges+num_specials}.\n'
                       f're-init the tokenizer with lower `explicit_n_vocab` in between {256+num_specials}'
                       f'(zero-merge) & {256+num_occured_merges+num_specials}(exactly-ran-out-of current corpus), '
                       f'or with enlarged corpus.')




def raise_disallowed_special_token(token: str) -> t.NoReturn:
    raise ValueError(
        f'disallowed specials {token!r} found in text.\n'
        f'expand `allowed_special` if you want to encode the disallowed marks into special tokens.\n'
        f'narrow `disallowed_special` if you want to encode the disallowed marks as plain.\n'
        f'you can expand `allowed_special` to "all" or narrow `disallowed_special` to (),'
        f'both will ignore specials and tokenize the text as plain'
    )


def encode_to_ints(s:str, encoding='utf-8') -> t.List[int]:
    return list( s.encode(encoding) )



class baseBBPETokenizer(Tokenizer):
    def __init__(
            self,
            name: str,
            pat_str: str = GPT4_TOKENIZER_REGEX,
            merge_ranks: dict[tuple[int, int], int] = {},
            special_marks: list[str] = [ENDOFTEXT, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, ENDOFPROMPT],
            explicit_n_vocab: int | None = None,
            **kwargs):

        self.name = name
        self.pat_str = pat_str
        self._merge_ranks = merge_ranks
        self._special_marks = special_marks
        # special marks 必须都能被 pat_str 切开，不然可能会导致 merge-generated in BPE
        assert all([ len(re.findall(pat_str, mark)) > 1 for mark in special_marks ])

        if merge_ranks: # 如果输入了非空的 merge_ranks
            # merge_ranks 的 RANK 应该是 256 至 MAX_RANK 的连续正整数: 递增 且 len(merge_ranks) = MAX_RANK - 255 且 首元素 = 256
            ranks_seq = list( merge_ranks.values() )
            assert check_monotonic(ranks_seq, mode='increase', strict=True) and len(ranks_seq) == ranks_seq[-1]-255 and ranks_seq[0] == 256

            # 可以直接 构建 vocab: token_ID --> bytes
            self._build_vocab()
            
            # 可以直接 注册 special_tokens，因为已经有 merge_ranks，无需 BPE train
            self._register_special_tokens()

            # 总的 vocab_size, 即 explicit_n_vocab 也随之 确定。不过若输入了 explicit_n_vocab，检查是否和 merge+special 匹配
            if explicit_n_vocab:
                assert explicit_n_vocab == 256 + len(merge_ranks) + len(special_marks) # 总 vocab size 应该等于 256 + merge tokens size + n_special_tokens
            self.explicit_n_vocab = 256 + len(merge_ranks) + len(special_marks)

        else: # 如果 没有输入非空的 merge_ranks
            # 那么需要 run BPE train process to build merges_ranks forrest. corpus text 将随后输入，但在这里可以 确定 number of merges
            if isinstance(explicit_n_vocab, int): # 如果输入了 explicit_n_vocab
                assert explicit_n_vocab >= 256 + len(special_marks), \
                    f'pretrained merge_ranks forrest empty.\n' + 'input explicit_n_vocab (shall be at least greater ' + \
                    f'than 256+{len(special_marks)}(num_special_marks)={256+len(special_marks)}.\n' + \
                    f'e.g, GPT2 tokenizer has explicit_n_vocab as 50257, with 1 special marks and 50000 merges.'
                
                # 需要执行 merge 的次数。但不一定能执行到，因为可能存在 run out of corpus 的情况（即所有 corpus 被merge到一起了）。此时 raise error
                self._num_merges = explicit_n_vocab - 256 - len(special_marks)


    def _build_vocab(self):
        # vocab: token_ID --> bytes
        assert hasattr(self, "_merge_ranks")
        self._vocab = {i: bytes([i]) for i in range(256)} # initialize 0 - 255 --> bytes
        for (L_int, R_int), merged_int in self._merge_ranks.items():
            self._vocab[merged_int] = self._vocab[L_int] + self._vocab[R_int] # two bytes concatenate


    def _register_special_tokens(self):
        assert hasattr(self, "_merge_ranks") and hasattr(self, "_special_marks")
        # vocab key 是 int, value 是 bytes
        # inverse special tokens key 是 int, value 是 str
        # special tokens key 是 str, value 是 int
        # 不统一，是因为 special tokens 要保持 str, 以便作 正则分割操作
        self.special_tokens: dict[str, int] = {} # speical mark str --> special token id

        # special tokens 的 token ID 应该 紧接着 merge_ranks 的 MAX RANK，即 MAX_RANK + 1 开始
        # 这样 所有tokens的 mapping value 应该是 0 至 explicit_n_vocab-1 = 256 + len(merge_ranks) + len(special_marks) - 1 的连续正整数
        # 所以 注册 special_tokens 的工作应该在 获得 有效的 merge_ranks 之后
        if self._merge_ranks:
            SPECIAL_RANK_START = max( self._merge_ranks.values() ) + 1 # special starts from MAX_MERGE_RANK + 1
        else:
            import warnings
            warnings.warn(
                f'merge_ranks is Empty. now register special tokens right after 0 - 255 bytes.\n'
                f'run valid BPE train process to build merge_ranks forrest before register special tokens')
            SPECIAL_RANK_START = 256 # special starts from 255 + 1

        for i, sp_mark in enumerate(self._special_marks):
            self.special_tokens[sp_mark] = SPECIAL_RANK_START + i
        
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()} # special token id --> special mark str


    def _prepare_train(self, num_merges:int|None = None, *args, **kwargs):
        if not hasattr(self, '_num_merges'): # if _num_merges not set, must input num_merges here
            assert isinstance(num_merges, int) and num_merges >= 0, f'num merges not set. must input num_merges >= 0.'
            self._num_merges = num_merges
        elif isinstance(num_merges, int) and self._num_merges != num_merges: # if _num_merges already set, but not equal to num_merges
            # warning. used pre-set _num_merges
            import warnings
            warnings.warn(
                f'input `num_merges`{num_merges} is not consistent with `explicit_n_vocab` from initialization.\n'
                f'merge times derived from `explicit_n_vocab` shall be `explicit_n_vocab`-num_specials-256 = {self._num_merges}.\n'
                f'ignore `num_merges`, use `explicit_n_vocab` to run BPE.')
        
        self._merge_ranks: dict[tuple[int, int], int] = {} # 初始化 _merge_ranks
        self._vocab: dict[int, bytes] = {i:bytes([i]) for i in range(256)} # 初始化 _vocab


    def _update_tokenizer(self, occur_most_pair:tuple[int, int], new_token:int, occurence:int|None):
        # rank = i: 0, ..., _num_mergs-1
        self._merge_ranks[occur_most_pair] = new_token
        self._vocab[new_token] = self._vocab[occur_most_pair[0]] + self._vocab[occur_most_pair[1]]

        if occurence: # occurence 不会是0
            print(f'merge process {len(self._merge_ranks)}/{self._num_merges}: {occur_most_pair} -> {new_token}'
                  f'[{self._vocab[new_token]}] had {occurence} occurences')


    def bpe_single_merge(self, rank:int, tokens_generator:t.Generator, verbose:bool=False):
        # rank 从 0 开始
        agg_p_counts:t.Dict[tuple[int, int], int] = {}
        stored_tokens = []
        for tokens in tokens_generator:
            # 对于最多只有 1个token 的 tokens(即all merged together)
            # 它从本次 merge开始，不会再贡献影响任何后续 p_counts, 也没有更新的必要
            if len(tokens) <= 1:
                continue
            get_pair_counts(tokens, agg_p_counts)
            stored_tokens.append( tokens )
        
        if not agg_p_counts:
            raise_run_out_corpus_error(rank, len(self._special_marks))
        
        occur_most_pair: tuple[int, int] = max(agg_p_counts, key=agg_p_counts.get)
        occurence: int|None = agg_p_counts[occur_most_pair] if verbose else None
        new_token: int = rank + 256
        del agg_p_counts

        yield occur_most_pair, occurence, new_token # first yield

        # yield remain as new tokens_generator
        for tokens in stored_tokens:
            yield merge_pair(tokens, occur_most_pair, new_token)
    

    def train_bpe(self, corpora:t.List[str]|str, num_merges:int|None = None, verbose:bool=False, *args, **kwargs):
        if isinstance(corpora, str):
            corpora = [corpora]
        
        corpus = ENDOFTEXT.join( corpora )
        self._prepare_train(num_merges)
        
        chunks_str: t.List[str] = re.findall(self.pat_str, corpus) # pre-split to list of string

        with ThreadPoolExecutor(max_workers=8) as e:
             yield_tokens = e.map(encode_to_ints, chunks_str) # a generator
        
        # <merge循环有前后依赖，所以不能并行>
        for i in range(self._num_merges):
            # rank = i: 0, ..., _num_mergs-1
            yield_output = self.bpe_single_merge(i, yield_tokens, verbose)
            occur_most_pair, occurence, new_token = next(yield_output) # first yield
            
            self._update_tokenizer(occur_most_pair, new_token, occurence)
            
            yield_tokens = yield_output # continue to yield tokens. update yield_tokens

        # explicit_n_vocab = num_actual_merges + 256 + num_specials
        # 循环正常结束时, i = _num_merges - 1,  num_actual_merges = _num_merges ---> n_vocab = i + 1 + 256 + num_specials
        # 循环提前结束时, i = num_actual_merges                                 ---> n_vocab = i + 256 + num_specials
        # 都等于最近的 new_token + 1 + num_specials, 除非极端情况下 _num_merges = 0
        # 所以还是用最稳妥的计算方式
        self.explicit_n_vocab = 256 + self._num_merges + len(self._special_marks)
        self._register_special_tokens()


    def _encode_chunk(self, tokens:t.List[int]) -> t.List[int]:
        '''
        对 chunk(tokens) 作持续的 merge, 直至 无可merge
        '''
        while len(tokens) > 1: # 在循环体内不断更新 tokens
            p_counts: t.Dict[tuple[int, int], int] = get_pair_counts(tokens)

            min_rank_pair: tuple[int, int] = min(p_counts, key=lambda p: self._merge_ranks.get(p, float('inf')))

            if min_rank_pair not in self._merge_ranks: # 如果 求出 的 min_rank_pair 不在 _merge_ranks 里
                break
            # 更新 tokens
            tokens = merge_pair(tokens, min_rank_pair, self._merge_ranks[min_rank_pair])
        
        return tokens

    
    def encode_ordinary(self, text:str) -> t.List[int]:
        '''
        encoding text that ignores any special tokens
        无视任何 special tokens / marks, 纯粹从 utf-8 编码后的字节流 作持续不断的 merge, 以 tokenize
        '''
        # encode 方法必须先检查有无 special_tokens. special_tokens 可以空, 也可以不在 encode 中用到.
        assert hasattr(self, 'special_tokens') # 有 / 无 special_tokens 属性, 可以区分 tokenizer 的 merge_ranks-empty / merge_ranks not-generated
        # raw
        chunks_str: t.List[str] = re.findall(self.pat_str, text) # pre-split to list of string
        # initial tokens: 可多线程加速
        chunks_tokens: t.List[list[int]] = [list( chunk.encode('utf-8') ) for chunk in chunks_str] # list of int(0..255)_list

        encoded_output: t.List[int] = []
        for tokens in chunks_tokens:
            encoded_output.extend( self._encode_chunk(tokens) )
        
        return encoded_output


    def encode_special(
            self,
            text:str,
            allowed_special:t.Literal["all"] | t.AbstractSet[str]
            ) -> t.List[int]:
        '''
        encoding text that first mapping registered and allowed special marks to special_token_ID, then tokenize the rest
        输入的 allowed_special 和 注册的 special_tokens 取交集.
            交集内的 special 若出现在 text 中, 则mapping as 注册的 special token ID
            text 其他部分采用 encode_ordinary 方式 编码
        '''
        # 要求本 tokenizer 已完成 specials 注册. special_tokens is dict of {str: int}
        assert hasattr(self, 'special_tokens') # 有 / 无 special_tokens 属性, 可以区分 tokenizer 的 merge_ranks-empty / merge_ranks not-generated
        if allowed_special == 'all':
            specials = self.special_tokens
        else:
            # allowed_special 交集 registered_special  如果 allowed_special 为空, 那么 specials 也为 空
            specials = {k:v for k, v in self.special_tokens.items() if k in allowed_special} # dict of str: int

        if not specials: # 如果 specials 为 空
            return self.encode_ordinary(text)
        
        special_pat = '(' + '|'.join(re.escape(k) for k in specials) + ')'
        chunks = re.split(special_pat, text) # special tokens 从 text 中分离出来: list of str

        tokens = []
        for chunk in chunks:
            if chunk in specials: # 如果是 special mark, 直接匹配 注册的 special token ID
                tokens.append( self.special_tokens[chunk] )
            else: # 如果是 plain text, 使用 encode ordinary 编码成 tokens
                tokens.extend( self.encode_ordinary(chunk) )
        
        return tokens
    
    
    def encode(self, text:str,
               allowed_special:t.Literal["all"] | t.AbstractSet[str] = set(),
               disallowed_special:t.Literal["all"] | t.Collection[str] = "all"
               ) -> t.List[int]:
        '''
        按理来说, special tokens 是拿来控制 LLM 的, 不应该出现在 text 中。这里 额外处理special的逻辑与 OpenAI tiktoken 保持一致。
        即：如果在 text 中检测到 special tokens, 则 raise error。
        
        通过 allowed_special, disallowed_special 来 控制 special tokens 的粒度。allow 和 disallow 的区别在于是否 raise error:

        第一步 确定 disallowed specials, 以此 判断本次 encode 要不要 raise error: 若 text 中出现了 disallowed specials, 则 raise error; 否则进入第二步
        第二步 用 encode_special 方法来 encode text: 即 allowed specials 和 注册 specials 的交集会被 map to special token ID
        
        1. 确定 disallowed specials。若 text 里包含 disallowed specials, 则 raise error。不包含则进入下一步
            如何确定 disallowed specials?
                1. input arg disallowed_special = all: 意味着 该tokenizer 注册的 special tokens 减去 arg allowed_special, 就是 disallowed specials
                （此时若 arg allowed_special = all, 则 disallowed_special 为 空，即 没有 disallow 的 special。）
                2. input arg disallowed_special = (): 意味着 disallowed_special 为 空，即 没有 disallow 的 special。
                3. input arg disallowed_special = set of str marks: 意味着 disallowed_special 是一个 valid 集合
        
        2. 若在 第1步没有 raise error, 则采用 encode with special on text
        '''
        assert hasattr(self, 'special_tokens') # 有 / 无 special_tokens 属性, 可以区分 tokenizer 的 merge_ranks-empty / merge_ranks not-generated

        if allowed_special == "all":
            allowed_special = set( self.special_tokens ) # dict of {str: int} ---> set of str

        if disallowed_special == "all":
            disallowed_special = set( self.special_tokens ) - allowed_special # set - set ---> set of str

        if disallowed_special: # 如果到这里, disallowed_special 非空, 那么要对 text 作检测，保证其不出现 disallowed_special, 不然 raise error
            if not isinstance(disallowed_special, frozenset):
                disallowed_special = frozenset(disallowed_special) # set --> frozenset

            if match := _special_token_regex(disallowed_special).search(text):
                raise_disallowed_special_token(match.group())
        
        return self.encode_special(text, allowed_special)

    
    def decode(self, tokens:t.List[int], errors: str = "replace") -> str:
        assert hasattr(self, '_vocab')

        parts = []
        for idx in tokens:
            if idx in self._vocab:
                parts.append( self._vocab[idx] ) # append bytes
            elif idx in self.inverse_special_tokens:
                parts.append( self.inverse_special_tokens[idx].encode('utf-8') ) # append bytes
            else:
                raise ValueError(f'invalid index {idx} out-of-vocab')
        concat_bytes = b''.join( parts )
        # 容错：错误的字节序列使用一个 replacement char 来替代
        return concat_bytes.decode('utf-8', errors=errors)
    

    def save(self, f_path):
        '''
        保存 name
        保存 pat_str
        保存 special_marks
        保存 merge_ranks (只需要保存 keys 即可，因为 values 是从 256 的递增序列)
        ---> 即保存了一个 tokenizer 的全部信息
        '''
        assert f_path.endswith(".tok")
        with open(f_path, 'w') as f:
            # write the name of the tokenizer as version
            f.write(f"{self.name}\n")
            # write the split pattern
            f.write(f"{self.pat_str}\n")
            # write the special_marks: first line number of special marks, then each line with a special mark
            f.write(f"{len(self._special_marks)}\n")
            for mark in self._special_marks:
                f.write(f"{mark}\n")
            # write the merge_ranks' keys
            for L, R in self._merge_ranks:
                f.write(f"{L} {R}\n")

    
    def load(self, f_path):
        '''
        读取 name: line 1
        读取 pat_str: line 2
        读取 num_of_special_marks: line 3
        读取接下来 num_of_special_marks 行 --> special marks
        按序读取剩下所有行: pair tokens, 依次存入 merge_ranks[pair tokens] = 256 ++
        构建剩余所有其他, register special tokens / vocab / explicit_n_vocab
        '''
        assert f_path.endswith(".tok")
        # read .tok file
        self._special_marks = []
        self._merge_ranks = {}
        with open(f_path, 'r', encoding='utf-8') as f:
            # line 1: the name of the tokenizer as version
            self.name = f.readline().strip()
            # line 2: the split pattern
            self.pat_str = f.readline().strip()
            # line 3: the num_special
            num_special = int(f.readline().strip())
            # at line 4, next num_special lines
            for _ in range(num_special):
                self._special_marks.append( f.readline().strip() )
            # all remained lines as pair merged, if exists: split and store them in order
            for i, line in enumerate(f):
                L, R = map(int, line.split())
                self._merge_ranks[(L, R)] = 256 + i # i 从 0 开始, rank 从 256 开始

            try: # 循环结束时, i = num_lines_of_merge_ranks - 1 , explicit_n_vocab = 256 + num_merges + num_specials
                self.explicit_n_vocab = 257 + i + num_special
            except UnboundLocalError: # fix rare situation when no remained lines, that is no pair merged
                self.explicit_n_vocab = 256 + num_special

        
        # 构建 vocab: token_ID --> bytes
        self._build_vocab()

        # 注册 special_tokens
        self._register_special_tokens()

    
    def view(self, tmpsave_dir):
        # _vocab: int(0 至 MAX_MERGE_RANK) --> bytes
        # _merge_ranks: (int, int) --> merged_int(256 至 MAX_MERGE_RANK)
        # special_tokens: (str, int)
        assert hasattr(self, "_vocab") and hasattr(self, "_merge_ranks") and hasattr(self, "special_tokens")
        reverse_merge = {v:k for k, v in self._merge_ranks.items()} # merged_int --> (int, int)

        from .text_preprocess import render_bytes
        import os

        with open(os.path.join(tmpsave_dir, f'tmp_{self.name}.vocab'), 'w', encoding='utf-8') as f:
            # 首先打印 special marks:
            for mark, idx in self.special_tokens.items():
                f.write(f"[{mark}] {idx}\n")

            for idx, token in self._vocab.items():
                s = render_bytes(token)
                if idx < 256:
                    f.write(f"[{s}] {idx}\n")
                else:
                    L, R = reverse_merge[idx]
                    s_L, s_R = render_bytes(self._vocab[L]), render_bytes(self._vocab[R])
                    f.write(f"[{s_L}] {L} , [{s_R}] {R} -> [{s}] {idx}\n")


    @property
    def vocab_size(self) -> int:
        return self.explicit_n_vocab
    

    @property
    def eot_token(self) -> int:
        return self.special_tokens[ENDOFTEXT]


    @functools.cached_property
    def special_marks_set(self) -> set[str]:
        return set(self.special_tokens.keys())
    

    def is_special_token(self, token: int) -> bool:
        assert isinstance(token, int)
        return token in self.inverse_special_tokens
    












import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
from pathlib import Path
from ..file.parquet_io import yield_parquet_batch
from ..file.folder_op import clean_folder





def check_tokens_schema(tokens_schema):
    assert len(tokens_schema) == 1 and \
           isinstance(tokens_schema.field(0).type, pa.ListType) and \
           (tokens_schema.field(0).type.value_type.equals(pa.int32()) or \
            tokens_schema.field(0).type.value_type.equals(pa.int64())), \
f'''
tokens_schema must be as following:
pa.schema([
    pa.field(
        'tokens',
        pa.list_(
            pa.field('token', pa.int32())
        )
    )
])
'''




class bufferBBPETokenizer(baseBBPETokenizer):
    '''
    get_pair_counts 和 merge_pair 两个 work on tokens(list of integers) 的原子操作, 在超长的 chunks of tokens 上并发执行
    的收益非常低。由于 tokens 一般被切得很小, 故 这两个原子操作的计算密度不大, 而超长的 chunks of tokens 的并发数量太大，并发
    带来的开销完全抵消了其提升。
    真正的瓶颈在于 stored_tokens(暂存的tokens以merge top pair) 导致的内存瓶颈。超长的 stored_tokens 是 list of tokens, 长
    度非常长，单机内存很可能放不下。 应该首先处理这个内存瓶颈。-----> buffer it
    '''
    p_counts_schema = pa.schema([
        pa.field('L', pa.int32()),
        pa.field('R', pa.int32()),
        pa.field('counts', pa.int64()),
        ])
    
    
    tokens_schema = pa.schema([
        pa.field( 'tokens', pa.list_(pa.field('token', pa.int32())) ),
        ])


    def __init__(self, buffer_dir, **kwargs):
        super().__init__(**kwargs)
        self._buffer_dir = buffer_dir

    
    def text_to_tokens_pa_table(self, text):
        if not text.endswith(ENDOFTEXT):
            text = text + ENDOFTEXT
        chunks_str = re.findall(self.pat_str, text) # list of tokens(string)
        
        # list of list of integers(every list of integers as tokens)
        chunks_tokens = [encode_to_ints(chunk) for chunk in chunks_str]
        
        # 创建 pa table
        batch_table = pa.Table.from_pydict({self.tokens_schema[0].name: chunks_tokens}, self.tokens_schema)

        return batch_table


    def _set_buffer_env(self, buffer_size):

        os.makedirs(self._buffer_dir, exist_ok=True)
        # buffer_dir 里应该为 empty, 或有 tokens 和 p_counts 两个空文件夹
        assert os.listdir(self._buffer_dir) == [] or \
               set(os.listdir(self._buffer_dir) ) == set(['tokens', 'p_counts']),\
            f'buffer_dir shall only contains at most two folders: tokens & p_counts'
        
        buffer_tokens_dir = os.path.join(self._buffer_dir, 'tokens')
        os.makedirs(buffer_tokens_dir, exist_ok=True)
        assert os.listdir(buffer_tokens_dir) == [], f'buffer_dir/tokens {buffer_tokens_dir} shall be empty'
        
        buffer_pcounts_dir = os.path.join(self._buffer_dir, 'p_counts')
        os.makedirs(buffer_pcounts_dir, exist_ok=True)
        assert os.listdir(buffer_pcounts_dir) == [], f'buffer_dir/p_counts {buffer_pcounts_dir} shall be empty'

        self._buffer_tokens_dir = buffer_tokens_dir
        self._buffer_pcounts_dir = buffer_pcounts_dir
        self._buffer_size = buffer_size



    def _init_tokens_pqs(self, corpus_pqs, text_colnames) -> str:
        '''
        initialize tokens parquet files from corpus parquet files list to:
        [   /buffer_dir/tokens/0/tokens_pq_1
            /buffer_dir/tokens/0/tokens_pq_2
                      ...
            /buffer_dir/tokens/0/tokens_pq_m   ]
        '''
        os.makedirs( os.path.join(self._buffer_tokens_dir, '0'), exist_ok=True)
        init_tokens_pqs = []
        for corpus_pq in corpus_pqs:
            corpus_fname = os.path.basename(corpus_pq)
            init_tokens_pq = os.path.join(self._buffer_tokens_dir, '0', corpus_fname)

            if init_tokens_pq in init_tokens_pqs:
                raise FileExistsError(
                    f'input corpus parquet files cannot have duplicated file basename.\n'
                    f'now they are {corpus_pqs}.\n'
                    f'change the basename of input corpus file in parquet format, because the program '
                    f'generated same name for previous corpus in string format.')
            else:
                init_tokens_pqs.append(init_tokens_pq)
        
        for corpus_pq, text_col, init_tokens_pq in zip(corpus_pqs, text_colnames, init_tokens_pqs):
            corpus_batch_iter = yield_parquet_batch(corpus_pq, 8192, [text_col])

            with pq.ParquetWriter(init_tokens_pq, self.tokens_schema) as writer:
                for k, batch in enumerate(corpus_batch_iter):
                    try:
                        text = ENDOFTEXT.join( batch[text_col].to_pylist() )
                        # 创建 pa table
                        batch_table = self.text_to_tokens_pa_table(text)
                        writer.write_table(batch_table)
                    except:
                        raise RuntimeError(
                            f'convert & write corpus parquet {corpus_pq} to tokens parquet error in batch {k}'
                            )
        
        return init_tokens_pqs


    def _prepare_train(self, num_merges, corpora:t.List[str], columns:t.List[str|None], buffer_size):
        '''
        for i, corpus in enumerate(corpora):
            如果 corpus 是 endswith('.parquet') 的 pq file:
                从 corpus 中读取 batch of text, join/split 成 chunks 之后, 把每个 chunk encode 成 tokens(list of integers)
                把 tokens 写入到位于 buffer_dir 的 Parquet 文件。per tokens-chunk per row

                tokens-chunk 是 ListArray 类型 --> list of int32 named 'token' --> list_of_ints_type
                    变长的 List Array, field 是 token, dtype 为 int32(token的最大vocab size 十几万即可, int32值域足够)
                
                Parquet table 的 schema:
                    列名: tokens
                    column type: list_of_ints_type, which is pa.list_(pa.field('token', pa.int32()))
            
            如果 corpus 是普通 string 类文本:
                在 buffer_dir 中生成名为 `corpus.parquet` 的文件, 然后从 corpus.parquet 开始如上执行后续
        '''
        super()._prepare_train(num_merges)
        self._set_buffer_env(buffer_size)
        assert len(corpora) == len(columns), f"length of corpora must match with length of columns"

        corpus_pqs, text_colnames = [], []
        for i, corpus in enumerate(corpora):
            if not corpus.endswith('.parquet'):
                # 一列数据, 列名为 'text', 类型为 string
                corpus_pq_schema = pa.schema([ pa.field( 'text', pa.string() ) ])
                # 在 buffer_dir 中, 为第 i 个corpus自动生成唯一的pq
                corpus_pq = os.path.join(self._buffer_dir, f'corpus_{i}.parquet')
                with pq.ParquetWriter(corpus_pq, corpus_pq_schema) as writer:
                    writer.write_table(
                        pa.Table.from_pydict({"text":[corpus]}, corpus_pq_schema)
                        )
                corpus_pqs.append( corpus_pq )
                text_colnames.append( 'text' )
            elif os.path.exists(corpus) and columns[i]:
                corpus_pqs.append( corpus )
                text_colnames.append( columns[i] )
            else:
                raise FileNotFoundError(
                    f'corpus parquet file {corpus} not found or text_colname {columns[i]} not valid')
        
        init_tokens_pqs = self._init_tokens_pqs(corpus_pqs, text_colnames) # generate all init_tokens_pq file
        
        return init_tokens_pqs


    def _get_p_counts_pq(self, tokens_pq):

        yield_tokens:t.Generator = yield_parquet_batch(tokens_pq, self._buffer_size)
        part_p_counts:t.Dict[tuple[int, int], int] = {}

        for batch in yield_tokens: # 遍历读取当前 tokens_pq
            chunks_tokens = batch[self.tokens_schema[0].name].to_pylist()
            for tokens in chunks_tokens:
                get_pair_counts(tokens, part_p_counts)
        
        # buffer the part_p_counts. 虽然可以用 build_pa_table 直接用行数据创建, 但列数据的效率高很多
        datapoints = [ (l, r, count) for (l, r), count in part_p_counts.items() ]
        if datapoints:
            L_tokens, R_tokens, counts = zip(*datapoints)
        else:
            L_tokens, R_tokens, counts = [], [], []
        data = {
            self.p_counts_schema[0].name: list(L_tokens), # L: L_tokens
            self.p_counts_schema[1].name: list(R_tokens), # R: R_tokens
            self.p_counts_schema[2].name: list(counts), # counts: counts
            }
        del part_p_counts # 节省内存
        
        rank, token_fname = tokens_pq.split('/')[-2:] # 从 tokens_pq 中解析出 rank 和 tokens_pq 名字
        os.makedirs( os.path.join(self._buffer_pcounts_dir, rank), exist_ok=True) # 创建 本次 rank 的 buffer_pcounts_dir
        part_p_counts_pq = os.path.join(self._buffer_pcounts_dir, rank, token_fname)

        with pq.ParquetWriter(part_p_counts_pq, self.p_counts_schema) as writer:
            writer.write_table( pa.Table.from_pydict(data, self.p_counts_schema) )
        
        return part_p_counts_pq
    

    def _aggregate_p_counts(self, part_pcounts_pqs):
        p_counts_collect = [] # list of pa tables

        for p_counts_pq in part_pcounts_pqs:
            p_counts_collect.append( pq.read_table(p_counts_pq) )

        _p_counts = pa.concat_tables( p_counts_collect )
        L, R, counts = self.p_counts_schema[0].name, self.p_counts_schema[1].name, self.p_counts_schema[2].name
        agg_p_counts = _p_counts.group_by([L, R]).aggregate([(counts, 'sum')]) # counts 列 --> counts_sum 列

        return agg_p_counts, '_'.join([counts, 'sum'])


    def _get_occur_most(self, agg_p_counts, agg_colname):
        max_occur = pc.max(agg_p_counts[agg_colname]).as_py()
        filter_mask = pc.equal(agg_p_counts[agg_colname], max_occur)

        _row = agg_p_counts.filter(filter_mask).slice(0, 1)

        L, R = self.p_counts_schema[0].name, self.p_counts_schema[1].name
        occur_most_pair: tuple[int, int] = (_row[L][0].as_py(), _row[R][0].as_py())

        return occur_most_pair, max_occur

    
    def _merge_tokens_save(self, save_dir, to_merge_pair, new_token, tokens_pq):
        '''
        given tokens parquet file `tokens_pq`,
        merge `to_merge_pair` tokens to `new_token` inside every tokens chunk,
        then save result tokens chunks into a same-file-name parquet file to `save_dir`
        '''
        # 重新遍历读取当前 tokens_pq, merge 当前 tokens_pq 的 occur_most_pair, 缓存 merged tokens parquet file
        yield_tokens:t.Generator = yield_parquet_batch(tokens_pq, self._buffer_size)

        # merged result 也是下一个 merge rank 要读取的 parquet
        merged_tokens_pq = os.path.join(save_dir, os.path.basename(tokens_pq))
        
        with pq.ParquetWriter(merged_tokens_pq, self.tokens_schema) as writer:
            for batch in yield_tokens:
                chunks_tokens = batch[self.tokens_schema[0].name].to_pylist()
                merged_tokens = [ merge_pair(tokens, to_merge_pair, new_token) for tokens in chunks_tokens if len(tokens) > 1 ]
                new_batch = pa.RecordBatch.from_pydict({self.tokens_schema[0].name: merged_tokens}, self.tokens_schema)
                writer.write_batch( new_batch )
        
        return merged_tokens_pq



    def _next_tokens_pqs(self, tokens_pqs, occur_most_pair, new_token):
        # 在计算获得 tokens parquet for next merge_rank 时, 当前 tokens_pq 已经提炼出了 merge_info
        # 并更新了 tokenizer._merge_rank，使得其 + 1。所以 cur_dir_for_tokens_pq = len(_merge_rank) - 1
        # next_dir_for_tokens_pq = cur_dir_for_tokens_pq + 1 = len(cur_merge_rank)
        next_rank = len(self._merge_ranks)
        assert all([next_rank == int(Path(f).parent.name)+1 for f in tokens_pqs])
        
        next_dir_for_tokens = os.path.join(self._buffer_tokens_dir, f'{next_rank}')
        os.makedirs( next_dir_for_tokens, exist_ok=True)

        next_tokens_pqs = []
        for tokens_pq in tokens_pqs:
            print(f'merging occur-most pair tokens for {tokens_pq}')
            merged_tokens_pq = self._merge_tokens_save(
                next_dir_for_tokens,
                occur_most_pair, 
                new_token, 
                tokens_pq)
            
            next_tokens_pqs.append( merged_tokens_pq )
        
        return next_tokens_pqs # update tokens_pq



    def _get_merge_info(self, tokens_pqs):
        # rank = i: 0, ..., _num_mergs-1, 代表已经完成的 merge 次数
        rank = len(self._merge_ranks)
        # tokens_pqs 所在的 dir_name N 代表这些 tokens_pq 经过了 N 次 merge
        assert all([rank == int(Path(f).parent.name) for f in tokens_pqs])
        print(f'merge epoch {rank}')

        # compute pair_counts for every tokens parquet into p_counts parquet, and collect them
        print(f'collecting partial pair-counts')
        part_p_counts_pqs = []
        for tokens_pq in tokens_pqs:
            print(f'computing partial pair-counts for {tokens_pq}')

            if not os.path.exists(tokens_pq):
                raise FileNotFoundError(f'buffer parquet {tokens_pq} of merge_rank {rank} not found')
            
            part_p_counts_pqs.append( self._get_p_counts_pq(tokens_pq) )

        # aggregate pair counts to agg_p_counts(pa.Tabel)
        print(f'aggregating pair-counts')
        agg_p_counts, agg_colname = self._aggregate_p_counts(part_p_counts_pqs)
        if not agg_p_counts:
            raise_run_out_corpus_error(rank, len(self._special_marks))
        
        # obtain the pair with most occurrence
        print(f'calculating occur-most pair tokens')
        occur_most_pair, max_occurence = self._get_occur_most(agg_p_counts, agg_colname)
        del agg_p_counts

        return occur_most_pair, max_occurence



    def train_bpe(self,
                  corpora:t.List[str]|str,
                  text_columns:t.List[str|None] = ['text'], # 默认输入单个 pq file as corpus with text column `text`
                  buffer_size:int = 4194304, # max tokens-chunks in memory
                  keep_window:int = 10, # max reserved tokens_pq file in disk
                  num_merges:int|None = None,
                  verbose:bool = False,
                  *args, **kwargs):
        
        assert keep_window >= 0
        if isinstance(corpora, str):
            corpora = [corpora]
            text_columns = [None]
        
        tokens_pqs = self._prepare_train(num_merges, corpora, text_columns, buffer_size)

        for i in range(self._num_merges):
            # rank = i = len(self._merge_rank)

            occur_most_pair, max_occurence = self._get_merge_info(tokens_pqs)
            new_token, occurence = i + 256, max_occurence if verbose else None

            # update tokenizer: len(self._merge_rank) += 1
            self._update_tokenizer(occur_most_pair, new_token, occurence)

            # rank = i = len(self._merge_rank) - 1
            if i == self._num_merges - 1: # 完成最后一次 tokens merge 后, 即 i == _num_merges-1, 不需要继续 merge tokens
                break
            
            tokens_pqs = self._next_tokens_pqs(tokens_pqs, occur_most_pair, new_token)

            # keep the init and `keep_window` tokens/p_counts parquet file
            to_remove = i - keep_window
            if to_remove > 0:
                clean_folder( os.path.join(self._buffer_tokens_dir, f'{to_remove}') )
                clean_folder( os.path.join(self._buffer_pcounts_dir, f'{to_remove}') )

        self.explicit_n_vocab = 256 + self._num_merges + len(self._special_marks)
        self._register_special_tokens()












#TODO
class asyncBBPETokenizer(bufferBBPETokenizer):
    '''
    bpe_single_merge: 
    前半部分：生产者不断生产 tokens(list of ints)
                消费者不断消费(一.把tokens写入disk stored_tokens; 二.计算partial_p_counts,累加到 agg_p_counts)
    tokens_generator  --tokens-->  append in stored_tokens disk
                        --tokens-->  get_pair_counts --> partial_p_counts --> aggregate in agg_p_counts

    tokens_generator替换为一个 fetch-tokens 协程:
    协程 tokens_generator:
        协程开始
        await fetch_next_tokens # 暂停直到 next tokens fetched
        协程恢复
        yield next_tokens # 吐出 next tokens

    stored_tokens:
    协程 tokens_generator 一旦吐出 tokens, if len(tokens) > 1, write tokens to stored_tokens

    agg_p_counts:
    协程 tokens_generator 一旦吐出 tokens, if len(tokens) > 1, apply get_pair_counts on tokens, get p_counts, aggregate p_counts in agg_p_counts

    直到 fetch all tokens
    '''
    pass