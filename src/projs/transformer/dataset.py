from ...core.utils.text.text_preprocess import preprocess_space
from ...core.utils.text.vocabulize import Vocab
from ...core.utils.common.seq_operation import truncate_pad
from ...core.utils.text.string_segment import sentence_segment_greedy
import torch
import typing as t
import re
import random



def tokenize_seq2seqText(
        raw_text:str, # raw text data
        sample_separator:str, # 样本之间的分隔符, 默认为 \n
        srctgt_separator:str, # source seq 和 target seq 之间的分隔符, 默认为 \t
        # params for sentence_tokenizer
        src_glossary:t.Dict|None, # 切割 source seq 用的 glossary. None 代表 word 即 token
        tgt_glossary:t.Dict|None, # 切割 target seq 用的 glossary. None 代表 word 即 token
        UNK_token:str, # 切割 source/target seq 时, 代表 无法识别切割的部分的 unkown token. src 和 tgt 用一个 UNK 就行
        # 设定 文本统一预处理 的格式
        need_lower=True, # 是否小写化
        separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~', # 独立标点符号集
        # 样本量
        num_examples=None,
        ):
    """
    inputs:
        raw_text: seq2seq text dat file, samples separated by arg sample_separator, src & tgt separated by arg srctgt_separator
        sample_separator: separator between samples (default as \n)
        srctgt_separator: separator inside a sample (default as \t)
        
        src_glossary: 切割 source seq 用的 glossary, None代表使用word整体作为source token. EOW 已经包含其中
        tgt_glossary: 切割 target seq 用的 glossary, None代表使用word整体作为target token. EOW 已经包含其中
        UNK_token: 切割 source/target seq 时, 代表 无法识别切割的部分的 unkown token 
        
        need_lower=True, 是否小写化
        separate_puncs=',.!?', 独立标点符号集
        
        num_examples: max sample size to process

    returns: denoted as source, target
        source: 2D list, each element is a datapoint, which is a list of source token sequence
        target: 2D list, each element is a datapoint, which is a list of target token sequence correspondingly
    
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
            break
        try:
            src_sentence, tgt_sentence = line.split(seq_sep) # 每一行分成两部分, 前半是 source sentence，后半是 target sentence
            # source list append tokenized src_seq token list
            source.append( sentence_segment_greedy(src_sentence, src_glossary, UNK_token, flatten=True)[0] )
            # target list append tokenized src_seq token list
            target.append( sentence_segment_greedy(tgt_sentence, tgt_glossary, UNK_token, flatten=True)[0] )
        except ValueError:
            raise ValueError(f"line {i+1} of text unpack wrong. line text as {line[:100]}")

    return source, target








def tensorize_tokens(tokens, vocab, num_steps):
    """
    inputs:
        tokens: 2D list of tokens: list of list of tokens
        vocab: the Vocab of input lines who can map token(word, char) to index
        num_steps: hyperparams to identify the length of sequences by truncating if too long or padding if too short

    returns: denoted as array, valid_len
        array: 2-dim tensor, shape as ( sample_size, num_steps )int64
        valid_len: 1-dim tensor, shape as ( sample_size, )int32, whose elements are the counts of non-padding tokens of lines
    
    explains:
        map tokens of input lines into indices according to input vocab, and set the sequence length
        将输入的词元序列lines映射成数字序列, 并设定num_steps(sequence length)序列长度. 不足num_steps的pad, 超出的剪掉. 
    """
    # id映射: lines 2D list, l & vocab[l] list. 在每个 line 末尾添加 vocab['<eos>']. 注意这里不能用append, append 在 列表生成式里返回None
    tokens = [vocab[l] + [vocab['<eos>']] for l in tokens]

    # tensor化 truncate_pad 之后的token序列, 默认就是int64
    array = torch.tensor([ truncate_pad(l, num_steps, vocab['<pad>']) for l in tokens], dtype=torch.int64)
    
    # 求出每个样本序列的valid length, 即token不是pad的个数, dtype 用 int32 节省空间
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)

    return array, valid_len






def build_seq2seqDataset(raw_text:str,
                         num_steps:int,
                         src_vocab:Vocab,
                         tgt_vocab:Vocab,
                         sample_separator:str,
                         srctgt_separator:str,
                         need_lower:bool=True,
                         separate_puncs:str='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
                         num_examples:int|None=None,
                         ):
    """
    inputs:
        raw_text: seq2seq text file
        num_steps: hyperparams to identify the length of sequences by truncating if too long or padding if too short
        src_vocab: vocab for source language
        tgt_vocab: vocab for target language
        sample_separator: separator for datapoints
        srctgt_separator: separator for source & target sequences
        need_lower: if lower the raw text
        separate_puncs: punctuations that shall be seen as independent tokens
        num_examples: total sample size if given. None to read all

    returns: denoted as tuple of tensors(tensor dataset), tuple of vocabs
        tensor dataset:
            1. source seqs, shape (num_examples, num_steps)
            2. source seqs valid(not-pad) lens, shape (num_examples, )
            3. target seqs, with shape (num_examples, num_steps)
            4. target seqs valid(not-pad) lens, with shape (num_examples, )

    explains:
        输入 seq2seq 数据集, 和 src/tgt 语言的 vocab; 输入num_steps; 以及其他参数
        返回 datasets, 其中tensors是(src数据集int64, src有效长度集int32, tgt数据集int64, tgt有效长度集int32)
    """
    # 确保两个字典使用同一个 unk_token
    assert src_vocab.to_tokens(src_vocab.unk) == tgt_vocab.to_tokens(tgt_vocab.unk), f'different unk_token in src/tgt vocabs'

    unk_token = src_vocab.to_tokens(src_vocab.unk)

    # tokenize src / tgt sentences. source 和 target 是 2D list of tokens
    
    source, target = tokenize_seq2seqText(
        raw_text,
        sample_separator,
        srctgt_separator,
        {'tokens':src_vocab.tokens, 'EOW_token':src_vocab.to_tokens(src_vocab.eow)} if src_vocab.to_tokens(src_vocab.eow) else None,
        {'tokens':tgt_vocab.tokens, 'EOW_token':tgt_vocab.to_tokens(tgt_vocab.eow)} if tgt_vocab.to_tokens(tgt_vocab.eow) else None,
        unk_token,
        need_lower,
        separate_puncs,
        num_examples)
    
    # 制作 tensor dataset
    src_array, src_valid_len = tensorize_tokens(source, src_vocab, num_steps)# all src data, shapes (num_examples, num_stpes), (num_examples,)
    tgt_array, tgt_valid_len = tensorize_tokens(target, tgt_vocab, num_steps)# all tgt data, shapes (num_examples, num_stpes), (num_examples,)

    return (src_array, src_valid_len, tgt_array, tgt_valid_len)
















class seq2seqDataset(torch.utils.data.Dataset):
    def __init__(self,
                 text_path,
                 num_steps,
                 src_vocab_path,
                 tgt_vocab_path,
                 sample_separator='\n',
                 srctgt_separator='\t',
                 need_lower=True,
                 separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
                 num_examples:int|None=None):
        
        super().__init__()
        src_vocab, tgt_vocab = Vocab(), Vocab()
        src_vocab.load(src_vocab_path)
        tgt_vocab.load(tgt_vocab_path)

        with open(text_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        
        X, X_valid_lens, Y, Y_valid_lens = build_seq2seqDataset(
            raw_text, num_steps, src_vocab, tgt_vocab, sample_separator, srctgt_separator, need_lower, separate_puncs, num_examples)
        
        # X 是 source data 的 (batch_size, num_steps), Y 是 target data 的 (batch_size, num_steps)
        # X_valid_lens 是 source 的 (batch_size,), Y_valid_lens 是 target 的 (batch_size,)
        
        bos = torch.tensor( [tgt_vocab['<bos>']] * Y.shape[0], device=Y.device).reshape(-1, 1)
        # encoder 只对 source data 作深度表征, 故只需要 source data: X 和 X_valid_lens
        # decoder 需要结合source信息, 对 target data 作 timestep 0 -> num_steps-1 至 1 -> num_steps 的预测, 故需要
        # 步骤1 对 target data timestep 0 -> num_steps-1 作深度表征, 步骤2 传入 source data 信息. 步骤3 给出 target data timestep 1 -> num_steps
        Y_frontshift1 = torch.cat([bos, Y[:, :-1]], dim=1) # target data timestep 0 -> num_steps-1 , 即 decoder的输入之一. 命名为 Y_frontshift1

        self._net_inputs = (X, Y_frontshift1, X_valid_lens) # 输入给 transformer network
        self._loss_inputs = (Y, Y_valid_lens) # 输入给 loss
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
    
    def __getitem__(self, index):
        return (tuple(tensor[index] for tensor in self._net_inputs),
                tuple(tensor[index] for tensor in self._loss_inputs))
    
    def __len__(self):
        return len(self._net_inputs[0])

    @property
    def src_vocab(self):
        return self._src_vocab
    
    @property
    def tgt_vocab(self):
        return self._tgt_vocab

