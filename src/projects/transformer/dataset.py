from src.utils.text.text_preprocess import preprocess_space
from src.utils.text.vocabulize import Vocab
from src.core.data.assemble import truncate_pad
from src.utils.text.string_segment import sentence_segment_greedy
import torch
import typing as t
import re
import random



def segment_seq2seqText(
        raw_text:str, # raw text data
        sample_separator:str, # 样本之间的分隔符, 默认为 \n
        srctgt_separator:str, # source seq 和 target seq 之间的分隔符, 默认为 \t
        # params for sentence_tokenizer
        glossary:t.Dict|None, # 切割 source/target seq 用的 glossary. None 代表 word 即 token
        UNK_token:str, # 切割 source/target seq 时, 代表 无法识别切割的部分的 unkown token
        need_lower=True, # 是否小写化
        separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~', # 独立标点符号集
        num_examples=None,
        ):
    """
    inputs:
        raw_text: seq2seq text dat file, samples separated by arg sample_separator, src & tgt separated by arg srctgt_separator
        sample_separator: separator between samples (default as \n)
        srctgt_separator: separator inside a sample (default as \t)
        glossary: 切割 source/target seq 用的 glossary, None代表使用word整体作为source token. EOW 已经包含其中
        UNK_token: 切割 source/target seq 时, 代表 无法识别切割的部分的 unkown token 
        need_lower=True, 是否小写化
        separate_puncs=',.!?', 独立标点符号集
        num_examples: max sample size to process
    returns: denoted as source, target
        source: 2D list, each element is a datapoint, which is a list of source token sequence
        target: 2D list, each element is a datapoint, which is a list of target token sequence correspondingly
        sample_size: int
    explains:
        fisrt pre-process translation seq2seq text data, then split it into source token 2-D list and target token 2-D list
        首先预处理翻译数据集, 然后将它分拆成 source 词元序列们 和 对应的target词元序列们
    """
    # 因为 src 和 tgt 之间由 \t 分隔, 行之间由 \n 分隔, 为了在 normalize 特殊空白字符后 保留 样本间以及样本内部的分隔符号, 故用特殊符号替换
    randint = random.randint(10000, 99999)
    sample_sep, seq_sep = f'{randint}samplesep{randint}', f'{randint}datalabelsep{randint}'
    assert sample_sep not in raw_text, f'temp sample separator exists in raw text.'
    assert seq_sep not in raw_text, f'temp source/target sequence separator exists in raw text.'

    # 替换 \n 和 \t 以顺利 normalize 其他空白字符
    raw_text = re.sub(sample_separator, sample_sep, raw_text) # 用新的 sample_sep 替换原来的 sample_separator
    raw_text = re.sub(srctgt_separator, seq_sep, raw_text) # 用新的 seq_sep 替换原来的 srctgt_separator
    # 统一预处理: 替换不间断空格为单空格, trim首尾空格, 保证文字和 圈定的标点符号 之间有 单空格, normalize 空白 到 单空格
    text = preprocess_space(raw_text, need_lower=need_lower, separate_puncs=separate_puncs, normalize_whitespace=True)

    source, target = [], []
    for i, line in enumerate(text.split(sample_sep)): # 大文本按行分隔
        if (num_examples and i >= num_examples) or (line == ''): # 当有最大样本数限制, 且循环已经收集足够样本时, 抑或是读到空行时, 跳出循环
            i -= 1 # 为了统计样本总量, 循环结束后要用到i. 如果循环在这里退出, i是多加了一个1(相比正常结束)
            break
        try:
            src_sentence, tgt_sentence = line.split(seq_sep) # 每一行分成两部分, 前半是 source sentence，后半是 target sentence
            # source list append tokenized src_seq token list
            source.append( sentence_segment_greedy(src_sentence, glossary, UNK_token, flatten=True)[0] )
            # target list append tokenized src_seq token list
            target.append( sentence_segment_greedy(tgt_sentence, glossary, UNK_token, flatten=True)[0] )
        except ValueError:
            raise ValueError(f"line {i+1} of text unpack wrong. line text as {line[:100]}")

    return source, target, i+1


def tensorize_tokens(subwords, vocab, context_size):
    """
    inputs:
        subwords: 2D list of subwords: list of list of subwords
        vocab: the Vocab of input lines who can map subwords(word, char) to tokens(integer)
        context_size: hyperparams to identify the length of sequences by truncating if too long or padding if too short
    returns: denoted as tensor, valid_len
        array: 2-dim tensor, shape as ( sample_size, context_size )int64
        valid_len: 1-dim tensor, shape as ( sample_size, )int32, whose elements are the counts of non-padding tokens of lines
    explains:
        map tokens of input lines into indices according to input vocab, and set the sequence length
        将输入的词元序列lines映射成数字序列, 并设定context_size(sequence length)序列长度. 不足context_size的pad, 超出的剪掉. 
    """
    # id映射: lines 2D list, l & vocab[l] list. 在每个 line 末尾添加 vocab['<eos>']. 注意这里不能用append, append 在 列表生成式里返回None
    tokens = [vocab[l] + [vocab['<eos>']] for l in subwords]
    array = torch.tensor([ truncate_pad(l, context_size, vocab['<pad>']) for l in tokens], dtype=torch.int64) #[B, context_size]
    valid_lens = (array != vocab['<pad>']).type(torch.int64).sum(1) #[B, ]

    return array, valid_lens


class seq2seqDataset(torch.utils.data.Dataset):
    def __init__(self,
                 text_path,
                 context_size,
                 vocab_path,
                 sample_separator='\n',
                 srctgt_separator='\t',
                 need_lower=True,
                 separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
                 num_examples:int|None=None):
        super().__init__()
        with open(text_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        self._vocab = Vocab()
        self._vocab.load(vocab_path)

        # source 和 target 是 2D list of subwords
        source, target, self.sample_size = segment_seq2seqText(
            raw_text,
            sample_separator,
            srctgt_separator,
            self._vocab.to_glossary(),
            self._vocab.to_subwords(self._vocab.unk),
            need_lower,
            separate_puncs,
            num_examples)
        self.src, self.src_valid_lens = tensorize_tokens(source, self._vocab, context_size)# (sample_size, context_size), (sample_size,)
        self.tgt, self.tgt_valid_lens = tensorize_tokens(target, self._vocab, context_size)# (sample_size, context_size), (sample_size,)

    def __getitem__(self, index):
        return self.src[index], self.src_valid_lens[index], self.tgt[index], self.tgt_valid_lens[index]
    
    def __len__(self):
        return self.sample_size

    @property
    def vocab(self):
        return self._vocab

