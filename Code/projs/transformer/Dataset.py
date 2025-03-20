from ...Utils.Text.TextPreprocess import preprocess_space
from ...Utils.Text.Vocabulize import Vocab
from ...Utils.Common.SeqOperation import truncate_pad
from ...Utils.Text.Tokenize import line_tokenize_simple
import torch
import typing as t

def read_text2str(path) -> str:
    """
    inputs: path
        path: str path of the text data
    
    returns: a str object
        read the whole text into a single str object
    
    explains:
        载入文本到一个str对象, 所有文本都读入内存
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read() # 文本对象.read()方法：将整个文本对象读进一个str对象

def tokenize_seq2seq(
        text,
        sentence_tokenizer:t.Callable,
        symbols:t.List[str] | t.Set[str] | None = None,
        num_examples=None):
    """
    inputs: text, num_example(optional)
        text: str object with \n to seperate lines, and each line consists of 'source_seq\ttarget_seq'
        sentence_tokenizer: callablem function, take a sentence as input, return a corresponding tokenized list of string as output
        symbols: the token symbols vocab to tokenize text
        num_examples: max sample size to read into memory
    
    returns: denoted as source, target
        source: 2D list, each element is a list of source token sequence
        target: 2D list, each element is a list of target token sequence correspondingly
    
    explains:
        process translation text data, split it into source token sequence and target token sequence
        处理翻译数据集, 返回source词元序列们和对应的target词元序列. 可以设定样本量
    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')): # 大文本按行分隔
        if num_examples and i >= num_examples: # 当有最大样本数限制, 且循环已经收集足够样本时, 跳出循环
            break
        try:
            src_sentence, tgt_sentence = line.split('\t') # 每一行按制表符分成两部分, 前半是 source sentence，后半是 target sentence
            source.append( sentence_tokenizer(src_sentence, symbols) ) # source list append tokenized 英文序列 token list
            target.append( sentence_tokenizer(tgt_sentence, symbols) ) # target list append tokenized 法文序列 token list
        except ValueError:
            raise ValueError(f"line {i+1} of text unpack wrong")

    return source, target

def build_tensorDataset(lines, vocab, num_steps):
    """
    inputs: lines, vocab, num_steps
        lines: 2D list of token(word, char) sequences
        vocab: the Vocab of input lines who can map token(word, char) to index
        num_steps: hyperparams to identify the length of sequences by truncating if too long or padding if too short

    returns: denoted as array, valid_len
        array: 2-dim tensor, shape as ( sample_size, num_steps )int64
        valid_len: 1-dim tensor, shape as ( sample_size, )int32, whose elements are the counts of non-padding tokens of lines
    
    explains:
        map tokens of input lines into indices according to input vocab, and set the sequence length
        将输入的词元序列lines映射成数字序列, 并设定num_steps(sequence length)序列长度
    """
    lines = [vocab[l] + [vocab['<eos>']] for l in lines] # id映射: lines 2D list, l & vocab[l] list. 在每个 line 末尾添加 vocab['<eos>']. 注意这里不能用append
    array = torch.tensor([ truncate_pad(l, num_steps, vocab['<pad>']) for l in lines], dtype=torch.int64) # tensor化 truncate_pad之后的token序列, 默认就是int64
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1) # 求出每个样本序列的valid length, 即token不是pad的个数, int32节省空间
    return array, valid_len

def build_dataset_vocab(path, num_steps, num_examples=None):
    """
    inputs: path, num_steps, num_examples(optional), is_train(optional)
        path: seq2seq text file path
        num_steps: hyperparams to identify the length of sequences by truncating if too long or padding if too short
        num_examples: total sample size if given. None to read all

    returns: denoted as tuple of tensors(tensor dataset), tuple of vocabs
        tensor dataset:
            1. source seqs, shape (num_examples, num_steps)
            2. source seqs valid(not-pad) lens, shape (num_examples, )
            3. target seqs, with shape (num_examples, num_steps)
            4. target seqs valid(not-pad) lens, with shape (num_examples, )
        src_vocab: vocab of source language corpus
        tgt_vocab: vocab of target language corpus
    
    explains:
        返回seq2seq翻译数据集, 其中tensors是(src数据集int64, src有效长度集int32, tgt数据集int64, tgt有效长度集int32)
        返回seq2seq翻译词汇表, (src词汇表, tgt词汇表)
    """
    raw_text = read_text2str(path) # read text

    # 预处理: 小写化, 替换不间断空格为单空格, 并trim首尾空格, 保证文字和,.!?符号之间有 单空格. 因为 src 和 tgt 之间由 \t 分隔, 所以不能 normalize 空白字符
    text = preprocess_space(raw_text, need_lower=True, separate_puncs=',.!?', normalize_whitespace=False)

    # 使用最简单的 line_tokenize_simple 作为 sentence tokenizer。symbols 为 None
    source, target = tokenize_seq2seq(text, line_tokenize_simple, None, num_examples) # tokenize src / tgt sentence. source 和 target 是 2D list of tokens

    # 制作 vocab
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>']) # 制作src词表
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>']) # 制作tgt词表

    # 制作 tensor dataset
    src_array, src_valid_len = build_tensorDataset(source, src_vocab, num_steps)# all src data, shapes (num_examples, num_stpes), (num_examples,)
    tgt_array, tgt_valid_len = build_tensorDataset(target, tgt_vocab, num_steps)# all tgt data, shapes (num_examples, num_stpes), (num_examples,)

    return (src_array, src_valid_len, tgt_array, tgt_valid_len), (src_vocab, tgt_vocab)



class seq2seqDataset(torch.utils.data.Dataset):
    def __init__(self, path, num_steps, num_examples=None):
        super().__init__()
        (X, X_valid_lens, Y, Y_valid_lens), (src_vocab, tgt_vocab) = build_dataset_vocab(path, num_steps, num_examples)
        # X 是 source data 的 (batch_size, num_steps), Y 是 target data 的 (batch_size, num_steps)
        
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

