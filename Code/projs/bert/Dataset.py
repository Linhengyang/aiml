import torch
import random
from ...Utils.Text.TextPreprocess import Vocab

def _read_wiki(data_dir):
    with open(data_dir, 'r') as f:
        lines = f.readlines() # list of strings. each string is several sentences(joined by ' . '), that is a paragraph
    paragraphs = [line.strip().lower().split(' . ') for line in lines if len(line.split(' . ')) > 2] # list of sentence lists. 句子数量要大于2
    random.shuffle(paragraphs)
    return paragraphs

def get_tokens_segments(tokens_a, tokens_b=None):
    # input token_a: list of tokens, [tk1, tk2,...,tkN]
    # input token_b if not none: list of tokens, [tk1, tk2,...,tkN]
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * len(tokens)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1]*(len(tokens_b) + 1)
    # 拼接两个tokens list和<cls><sep>, 并输出它们的segments
    return tokens, segments

def _get_next_sentence(sentence, next_sentence, paragraphs):
    # paragraphs: num_paragraphs 个 paragraph, 每个paragraph是list of sentences(a sentence is a list of tokens)
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs)) # 随机选一个paragraph, 再从该段随机选一句
        is_next = False
    return sentence, next_sentence, is_next # 输出1个sentence, next sentence, 是否连接flag

def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    # paragraph is a 2-D list: each list is a list of sentences. a sentence is a list of tokens. sentence和paragraph都不会是空list
    # paragraphs is a list of paragraph
    nsp_data_from_paragraph = []
    for i in range(len(paragraph)-1): # 当前paragraph
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i+1], paragraphs)
        # tokens_a和tokens_b要去掉可能存在于原文本的<cls><sep>
        tokens_a = [token for token in tokens_a if token not in ('<cls>', '<sep>')]
        tokens_b = [token for token in tokens_b if token not in ('<cls>', '<sep>')]
        if len(tokens_a) == 0 or len(tokens_b) == 0: # 如果任一句子去掉<cls>和<sep>后, 长度为0, 放弃
            continue
        if len(tokens_a) + len(tokens_b) + 3 > max_len: # 如果当前前后两句token数量+<cls>+2个<sep>数量超过max_len, 放弃
            continue # 即放弃拼接后过长的前后句对. 相当于truncate处理. 所以后面只需pad到这个max_len即可
        tokens, segments = get_tokens_segments(tokens_a, tokens_b) # 拼接后的token list, 和对应的segments
        nsp_data_from_paragraph.append( [tokens, segments, is_next] )
    return nsp_data_from_paragraph # 输出一个list, 元素是nsp单样本: [拼接后token list, segment list, 是否相邻flag]

# 对一个token list, 输入可mask的token positions(<cls>和<sep>不可被替换), 以及该token list要mask的token数量
# 输出替换后的token list, 替换的<mask>在list中的positions, 以及真实label token list
def _replace_mlm_tokens(tokens, candidate_mask_positions, num_mlm_masks, vocab):
    mlm_input_tokens = [token for token in tokens] # 将输入token list拷贝下来
    mask_positions_labels = [] # 记录(mask_position, mask_label) pair
    random.shuffle(candidate_mask_positions) # 打乱
    for mask_position in candidate_mask_positions:
        if len(mask_positions_labels) >= num_mlm_masks: # 当已经作了足够次数mask操作, 退出循环
            break
        mask_token = None
        if random.random() < 0.8: # 80%的概率, 用<mask>去mask
            mask_token = '<mask>'
        else:
            if random.random() < 0.5: # 10%的概率, 用随机token去mask
                mask_token = random.choice(vocab.idx_to_token)
            else: # 10%的概率, 用自身去mask
                mask_token = tokens[mask_position]
        mlm_input_tokens[mask_position] = mask_token # mask操作
        mask_positions_labels.append( (mask_position, tokens[mask_position]) ) # 记录被mask的token位置, 以及真实token
    return mlm_input_tokens, mask_positions_labels # 输出token list, list of (position_idx, true_label)

# 对一个token list, 计算可mask的token positions, 计算该token list要mask的token数量
# 然后作mask操作
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_mask_positions = []
    for i, token in enumerate(tokens):
        if token not in ('<cls>', '<sep>'):
            candidate_mask_positions.append(i)
    # 如果candidate_mask_positions还是[], 中断并打印错误信息
    assert len(candidate_mask_positions) > 0, f'invalid token list processing {tokens}'
    num_mlm_masks = max(1, round(len(tokens)*0.15))
    mlm_input_tokens, mask_positions_labels = _replace_mlm_tokens(tokens, candidate_mask_positions, num_mlm_masks, vocab)
    mask_positions_labels = sorted(mask_positions_labels, key=lambda x: x[0]) # 将mask_positions_labels按照positions从小到大排列
    mask_positions = [v[0] for v in mask_positions_labels]
    mask_labels = [v[1] for v in mask_positions_labels]
    return vocab[mlm_input_tokens], mask_positions, vocab[mask_labels] # 输出masked token_idx list, mask position list, mask true token_idx list

# 两个任务的输出组合
# [two_sentence_token_idx_list, mask_position_list, mask_label_token_idx_list], (two_sentence_segment_list, is_next_flag)
# 为了batch化处理, 将 two_sentence_token_idx_list/mask_position_list/mask_label_token_idx_list/two_sentence_segment_list 作pad到统一长度
# 根据mask_position_list/mask_label_token_idx_list, 同步生成一个mlm_weight_list, 对于pad元素权重设0
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_masks = round(max_len*0.15) # token list的最大长度乘以0.15
    all_tokens_idx, all_segments, valid_lens = [], [], []
    all_mask_positions, all_mlm_weights, all_mlm_labels_idx = [], [], []
    nsp_labels = []
    for tokens_idx, mask_positions, mlm_labels_idx, segments, if_next in examples:
        # pad tokens_idx
        pad_tokens_idx = tokens_idx + [vocab['<pad>'],]*(max_len - len(tokens_idx))
        all_tokens_idx.append( torch.tensor(pad_tokens_idx, dtype=torch.int64) )
        # pad segments
        pad_segments = segments + [0]*(max_len - len(segments))
        all_segments.append( torch.tensor(pad_segments, dtype=torch.int64) )
        # valid_lens
        valid_lens.append( torch.tensor(len(tokens_idx), dtype=torch.float32) )
        # pad mask_positions
        pad_mask_positions = mask_positions + [0]*(max_num_mlm_masks - len(mask_positions))
        all_mask_positions.append( torch.tensor(pad_mask_positions, dtype=torch.int64) )
        # pad mlm_labels_idx
        pad_mlm_labels_idx = mlm_labels_idx + [0]*(max_num_mlm_masks - len(mlm_labels_idx))
        all_mlm_labels_idx.append( torch.tensor(pad_mlm_labels_idx, dtype=torch.int64) )
        # weights for mlm_labels_idx: 0 for pad
        mlm_labels_weight = [1]*len(mlm_labels_idx) + [0]*(max_num_mlm_masks - len(mlm_labels_idx))
        all_mlm_weights.append( torch.tensor(mlm_labels_weight, dtype=torch.float32) )
        nsp_labels.append( torch.tensor(if_next, dtype=torch.int64) )
    return all_tokens_idx, all_segments, valid_lens, all_mask_positions, all_mlm_weights, all_mlm_labels_idx, nsp_labels


class wikitextDataset(torch.utils.data.Dataset):
    def __init__(self, fpath, max_len):
        super().__init__()
        # lower/按.分句, 没有对数字/其他符号
        paragraphs = _read_wiki(fpath) #list of lists of sentences. Now sentence is string, paragraphs is 2D list
        # tokenize. 空段落过滤, 段落中空字符串的句子过滤
        paragraphs = [[ line.split() for line in paragraph if len(line) > 0 ] for paragraph in paragraphs if len(paragraph) > 0] # 3D list
        # 为了建立vocab, 需要传入一个2D list
        sentences = [line for paragraph in paragraphs for line in paragraph]
        # 建立vocab
        self._vocab = Vocab(sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        nsp_sample = []
        for paragraph in paragraphs:
            # cur_nsp_sample: list of [拼接后token list, segment list, 是否相邻flag]
            cur_nsp_sample = _get_nsp_data_from_paragraph(paragraph, paragraphs, self._vocab, max_len)
            nsp_sample.extend(cur_nsp_sample)
        # 组装samples
        examples = []
        for tokens, segments, if_next in nsp_sample:
            # masked token_idx list, mask position list, mask true token_idx list
            tokens_idx, mask_positions, mlm_labels_idx = _get_mlm_data_from_tokens(tokens, self._vocab)
            examples.append( [tokens_idx, mask_positions, mlm_labels_idx, segments, if_next] )
        # 灌入dataset
        self.all_tokens_idx, self.all_segments, self.valid_lens, self.all_mask_positions, self.all_mlm_weights, self.all_mlm_labels_idx, self.nsp_labels = _pad_bert_inputs(examples, max_len, self._vocab)

    def __getitem__(self, index):
        # 单条样本
        # tokens_idx_list, segments_list, valid_len_int, mask_positions_list, mlm_weights_list, mlm_labels_idx_list, if_next_int
        return (tuple([self.all_tokens_idx[index], self.all_segments[index], self.valid_lens[index], self.all_mask_positions[index]]), 
                tuple([self.all_mlm_weights[index], self.all_mlm_labels_idx[index], self.nsp_labels[index]]))

    def __len__(self):
        return len(self.all_tokens_idx)
    
    @property
    def vocab(self):
        return self._vocab