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
from ..Common.SeqOperation import check_monotonic

## deprecated split-pattern for GPT2. use GPT4 version
# GPT2_TOKENIZER_REGEX = \
#     r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

GPT4_TOKENIZER_REGEX = \
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

ENDOFTEXT = '<|endoftext|>'
FIM_PREFIX = '<|fim_prefix|>'
FIM_MIDDLE = '<|fim_middle|>'
FIM_SUFFIX = '<|fim_suffix|>'
ENDOFPROMPT = '<|endofprompt|>'





class BBPETokenizer(Tokenizer):
    def __init__(
            self,
            name: str,
            pat_str: str = GPT4_TOKENIZER_REGEX,
            merge_ranks: dict[tuple[int, int], int] = {},
            special_marks: list[str] = [ENDOFTEXT, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, ENDOFPROMPT],
            explicit_n_vocab: int | None = None
            ):

        self.name = name
        self._pat_str = pat_str
        self._merge_ranks = merge_ranks
        self._special_marks = special_marks
        # special marks 必须都能被 pat_str 切开，不然可能会导致 merge-generated in BPE
        assert all([ len(re.findall(pat_str, mark)) > 1 for mark in special_marks ])

        if merge_ranks: # 如果输入了非空的 merge_ranks
            # merge_ranks 的 RANK 应该是 0 至 MAX_RANK 的连续正整数: 递增 且 len(merge_ranks) = MAX_RANK + 1 且 首元素 = 0
            ranks_seq = merge_ranks.values()
            assert check_monotonic(ranks_seq, mode='increase', strict=True) and len(ranks_seq) == ranks_seq[-1] + 1 and ranks_seq[0] == 0
            # 可以直接 注册 special_tokens，因为已经有 merge_ranks，无需 BPE train
            self.register_special_tokens()
            # 总的 vocab_size, 即 explicit_n_vocab 也随之 确定。不过若输入了 explicit_n_vocab，检查是否和 merge+special 匹配
            if explicit_n_vocab:
                assert explicit_n_vocab == len(merge_ranks) + len(special_marks) # 总 vocab size 应该等于 merge tokens size + n_special_tokens
            self.explicit_n_vocab = len(merge_ranks) + len(special_marks)
        else: # 如果 没有输入非空的 merge_ranks
            # 那么需要 run BPE train process to build merges_ranks forrest. corpus text 将随后输入，但在这里可以 确定 number of merges
            # 必须输入 explicit_n_vocab
            assert explicit_n_vocab >= 256 + len(special_marks), \
                f'pretrained merge_ranks forrest empty.\
                  input explicit_n_vocab (shall be at least greater than 256+{len(special_marks)}(num_special_marks))'
            self.explicit_n_vocab = explicit_n_vocab
            self._num_merges = explicit_n_vocab - 256 - len(special_marks)


    def register_special_tokens(self):
        self.special_tokens: dict[str, int] = {}
        # special tokens 的 token ID 应该 紧接着 merge_ranks 的 MAX RANK，即 MAX_RANK + 1 开始
        # 这样 所有tokens的 mapping value 应该是 0 至 explicit_n_vocab-1 = len(merge_ranks) + len(special_marks) - 1 的连续正整数

        # 所以 注册 special_tokens 的工作应该在 获得 有效的 merge_ranks 之后
        if self._merge_ranks:
            MAX_MERGE_RANK = self._merge_ranks.values()[-1]
            for i, sp_mark in enumerate(self._special_marks):
                self.special_tokens[sp_mark] = MAX_MERGE_RANK + i + 1
        else:
            raise RuntimeError(f'merge_ranks not build. run BPE train process to build merge_ranks forrest before register special tokens')
        
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}


    def train_bpe(self, corpus, verbose=False):
        #TODO

        # 得到 merge_ranks forrest 之后，注册 special_tokens
        self.register_special_tokens()


    def encode(self, string):
        #TODO
        pass


    def decode(self, indices):
        #TODO
        pass
    

    def save(self, f_prefix):
        #TODO
        pass
    

    def load(self, f_name):
        #TODO
        pass

    
    def view(self, dir_path):
        #TODO
        pass

