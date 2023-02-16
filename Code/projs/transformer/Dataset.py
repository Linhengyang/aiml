from ...Utils.Text.TextPreprocess import preprocess_space
from ...Utils.Text.TextPreprocess import Vocab
from ...Utils.Common.SeqOperation import truncate_pad
import torch

def read_text2str(path):
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

def tokenize_seq2seq(text, num_examples=None):
    """
    inputs: text, num_example(optional)
        text: str object with \n to seperate lines, and each line consists of 'source_seq\ttarget_seq'
        num_examples: max sample size to read into memory
    
    returns: denoted as source, target
        source: 2D list, each element is a list of source token sequence
        target: 2D list, each element is a list of target token sequence
    
    explains:
        process translation text data, split it into source token sequence and target token sequence
        处理翻译数据集, 返回source词元序列们和对应的target词元序列. 可以设定样本量
    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')): # 大文本按行分隔
        if num_examples and i >= num_examples: # 当有最大样本数限制, 且循环已经收集足够样本时, 跳出循环
            break
        parts = line.split('\t') # 每一行按制表符分开
        if len(parts) == 2:
            source.append(parts[0].split(' ')) # source列表添加英文序列token list
            target.append(parts[1].split(' ')) # target列表添加法文序列token list
    return source, target

def build_tensorDataset(lines, vocab, num_steps):
    """
    inputs: lines, vocab, num_steps
        lines: 2D list of token(word, char) sequences
        vocab: the Vocab of input lines who can map token(word, char) to index
        num_steps: hyperparams to identify the length of sequences by truncating if too long or padding if too short

    returns: denoted as array, valid_len
        array: 2-dim tensor, shape as ( sample_size, num_steps )
        valid_len: 1-dim tensor, shape as ( sample_size, ), whose elements are the counts of non-padding tokens of lines
    
    explains:
        map tokens of input lines into indices according to input vocab, and set the sequence length
        将输入的词元序列lines映射成数字序列, 并设定num_steps(sequence length)序列长度
    """
    lines = [vocab[l] for l in lines] # id映射
    lines = [ l + [vocab['<eos>']] for l in lines ] # 每句末尾加<eos>
    array = torch.tensor([ truncate_pad(l, num_steps, vocab['<pad>']) for l in lines]) # tensor化 truncate_pad之后的token 序列
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1) # 求出每个样本序列的valid length, 即token不是pad的个数
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
        返回seq2seq翻译数据集, 其中tensors是(src数据集, src有效长度集, tgt数据集, tgt有效长度集)
        返回seq2seq翻译词汇表, (src词汇表, tgt词汇表)
    """
    raw_text = read_text2str(path) # read text
    text = preprocess_space(raw_text) # preprocess
    source, target = tokenize_seq2seq(text, num_examples) # 词元化, 得到source语料序列和target语料序列
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>']) # 制作词表
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_tensorDataset(source, src_vocab, num_steps)# all src data, shapes (num_examples, num_stpes), (num_examples,)
    tgt_array, tgt_valid_len = build_tensorDataset(target, tgt_vocab, num_steps)# all tgt data, shapes (num_examples, num_stpes), (num_examples,)
    return (src_array, src_valid_len, tgt_array, tgt_valid_len), (src_vocab, tgt_vocab)

class seq2seqDataset(torch.utils.data.Dataset):
    def __init__(self, path, num_steps, num_examples=None):
        super().__init__()
        (X, X_valid_lens, Y, Y_valid_lens), (src_vocab, tgt_vocab) = build_dataset_vocab(path, num_steps, num_examples)
        bos = torch.tensor( [tgt_vocab['<bos>']] * Y.shape[0], device=Y.device).reshape(-1, 1)
        dec_X = torch.cat([bos, Y[:, :-1]], dim=1)
        self._net_inputs = (X, dec_X, X_valid_lens)
        self._loss_inputs = (Y, Y_valid_lens)
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

