import torch
import random
from ...Utils.Text.TextPreprocess import Vocab, subsample
from ...Utils.Common.SeqOperation import truncate_pad
from ...Compute.SamplingTools import NegativeSampling

# 从text中subsample, 得到corpus
# 从corpus中得到
#   centers list, 和对应的 contexts list of lists
# 对每个context, 得到其对应的negatives list
# 最终得到 centers list, context list of lists, negative list of lists
# 一个center 对应一个context list, 对应一个negative list

def get_centers_and_contexts(sentences, max_window_size, size_random=True):
    '''
    input:
        sentences: 2D list of lists of tokens
        max_window_size: center-context窗口的半边大小
        size_random: 窗口大小是否随机. 若不随机, 使用max_window_size作为窗口的半边大小
    output:
        centers: list, 所有tokens as centers
        contexts: list of lists, 所有center token对应的context tokens
    '''
    centers, contexts = [], []
    for line in sentences:
        if len(line) < 2:
            continue
        centers += line # 当前line的每一个token都是center
        for i in range(len(line)):
            if size_random:
                window_size = random.randint(1, max_window_size)
            else:
                window_size = max_window_size
            indices = list( range( max(0, i-window_size), min(len(line), i+window_size+1) ) )
            indices.remove(i)
            contexts.append([ line[idx] for idx in indices ])
    return centers, contexts


def build_skipgram_arrays(centers, contexts, negatives, pad_value):
    '''
    a record consists of
        center_word, context_words_list, negative_words_list
    concat last two, to
        center_word, context_negative_words_list
    pad 0 to context_negative_words_list to align all records
    create labels_list with same length as context_negative_words_list, 1 for context_word and 0 for others
    '''
    assert len(centers) == len(contexts) == len(negatives), 'inputs length mismatch'
    max_len = max( [len(c) + len(n) for _, c, n in zip(centers, contexts, negatives)] )
    contexts_negatives, masks, labels = [], [], []
    for center_word, context_words, negative_words in zip(centers, contexts, negatives):
        cur_contexts_negatives_words = context_words + negative_words # context words和negative words合并
        contexts_negatives.append( truncate_pad(cur_contexts_negatives_words, max_len, pad_value) ) # pad到最大长度
        cur_mask = [1]*len(cur_contexts_negatives_words) # mask 1 for context words和negative words
        masks.append( truncate_pad(cur_mask, max_len, 0) ) # 其他位置pad0
        cur_label = [1]*len(context_words) # mask 1 为context words
        labels.append( truncate_pad(cur_label, max_len, 0) ) # 其他位置pad0
    return centers, contexts_negatives, masks, labels


def build_cbow_arrays(contexts, centers, negatives, pad_value):
    '''
    a record consists of
        context_words_list, center_word, negative_words_list
    concat last two, to
        context_words_list, center_neg_words_list
    pad 0 to context_words_list to align all records
    create labels_list with same length as center_neg_words_list, 1 for center word and 0 for others
    '''
    assert len(centers) == len(contexts) == len(negatives), 'inputs length mismatch'
    max_len = max( [len(c) for c, _, _ in zip(contexts, centers, negatives)] )
    pad_contexts, center_negtives, masks, labels = [], [], [], []
    for context_words, center_word, negative_words in zip(contexts, centers, negatives):
        cur_center_negative_words = [center_word] + negative_words # center word和negative words合并
        center_negtives.append(cur_center_negative_words)
        pad_contexts.append( truncate_pad(context_words, max_len, pad_value) ) # pad到最大长度
        cur_mask = [1]*len(context_words) + [0]*(max_len-len(context_words)) # mask 1 for context words, 其他位置0
        masks.append(cur_mask)
        cur_label = [1] + [0]*len(negative_words) # center word label 1, negative words label 0
        labels.append(cur_label)
    return pad_contexts, center_negtives, masks, labels

class skipgramDataset(torch.utils.data.Dataset):
    def __init__(self, fpath):
        super().__init__()
        with open(fpath) as f:
            raw_text = f.read()
        sentences = [line.split() for line in raw_text.split('\n')]
        vocab = Vocab(sentences, min_freq=10)
        corpus, counter = subsample(sentences, vocab) #降采样后的token corpus和降采样前的token counter
        centers, contexts = get_centers_and_contexts(corpus, 2)
        # negative sampling
        population = vocab.to_tokens(list(range(1, len(vocab)))) # 采样的population是词汇表中所有词汇(去掉<unk>)
        sampling_weights = [counter[token]**0.75 for token in population]
        negativeSampler = NegativeSampling(population, sampling_weights, 5) # 5 negative labels for 1 positive label
        negatives = negativeSampler.sample(contexts)
        centers_idx, contexts_idx, negatives_idx = vocab[centers], vocab[contexts], vocab[negatives]
        centers, ctx_neg_list, masks, labels = build_skipgram_arrays(centers_idx, contexts_idx, negatives_idx, 0)
        self.centers = torch.tensor(centers).unsqueeze(1) # shape: (batch_size, 1)
        self.ctxs_negs = torch.tensor(ctx_neg_list) # shape: (batch_size, 2*(K+1)*mask_window_size)
        self.labels = torch.tensor(labels).type(torch.float32) # shape: (batch_size, 2*(K+1)*mask_window_size)
        self.masks = torch.tensor(masks) # shape: (batch_size, 2*(K+1)*mask_window_size)
        self._vocab = vocab
        self._counter = counter
    
    def __len__(self):
        return self.centers.shape[0]
    
    def __getitem__(self, index):
        return (self.centers[index], self.ctxs_negs[index], self.labels[index], self.masks[index])
    
    @property
    def vocab(self):
        return self._vocab
    
    @property
    def counter(self):
        return self._counter


class cbowDataset(torch.utils.data.Dataset):
    def __init__(self, fpath):
        super().__init__()
        with open(fpath) as f:
            raw_text = f.read()
        sentences = [line.split() for line in raw_text.split('\n')]
        vocab = Vocab(sentences, min_freq=10)
        corpus, counter = subsample(sentences, vocab) #降采样后的token corpus和降采样前的token counter
        centers, contexts = get_centers_and_contexts(corpus, 2)
        # negative sampling
        population = vocab.to_tokens(list(range(1, len(vocab)))) # 采样的population是词汇表中所有词汇(去掉<unk>)
        sampling_weights = [counter[token]**0.75 for token in population]
        negativeSampler = NegativeSampling(population, sampling_weights, 5) # 5 negative labels for 1 positive label
        negatives = negativeSampler.sample(centers) # CBOW模型中, centers是目标vector, 对其负采样
        centers_idx, contexts_idx, negatives_idx = vocab[centers], vocab[contexts], vocab[negatives]
        pad_contexts, center_negtives, masks, labels = build_cbow_arrays(contexts_idx, centers_idx, negatives_idx, 0)
        self.contexts = torch.tensor(pad_contexts) # shape: (batch_size, 2*mask_window_size)
        self.center_negatives = torch.tensor(center_negtives) # shape: (batch_size, K+1)
        self.labels = torch.tensor(labels).type(torch.float32) # shape: (batch_size, K+1)
        self.masks = torch.tensor(masks) # shape: (batch_size, K+1)
        self._vocab = vocab
        self._counter = counter
    
    def __len__(self):
        return self.contexts.shape[0]
    
    def __getitem__(self, index):
        return (self.contexts[index], self.center_negatives[index], self.labels[index], self.masks[index])
    
    @property
    def vocab(self):
        return self._vocab
    
    @property
    def counter(self):
        return self._counter