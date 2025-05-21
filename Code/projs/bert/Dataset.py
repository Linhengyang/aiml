import torch
import random
import copy
import pandas as pd
from ...Utils.Text.Vocabulize import Vocab
from ...Utils.Text.Tokenize import line_tokenize_greedy
from ...Utils.Common.SeqOperation import truncate_pad




def _read_shuffle(parquet_fpath):

    df = pd.read_parquet(parquet_fpath)
    
    lines = df['text'].tolist() # list of strings. each string maybe several sentences(joined by ' . '), that is a paragraph

    # 2-D list. lists of list of at least 2 sentences(filter)
    corpus = [line.strip().lower().split(' . ') for line in lines if ' . ' in line]
    random.shuffle(corpus) # 段落之间打乱

    return corpus





def _preprocess_tokenize(corpus, glossary, UNK_token,
                         need_lower=True, separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|'):
    
    # corpus: 2-D list

    tokens_corpus = []

    for paragraph in corpus:
        # paragraph: 1-D list of at least two sentences
        if not paragraph:
            continue

        _tokens_paragraph = []
        
        for sentence in paragraph:
            if sentence:
                tokens = line_tokenize_greedy(sentence, glossary, UNK_token, flatten=True,
                                              need_preprocess=True, need_lower=need_lower, separate_puncs=separate_puncs)[0]
                _tokens_paragraph.append( tokens )
        

        tokens_corpus.append( _tokens_paragraph )


    return tokens_corpus







def _concate_tokenlist_pair(tokens_a, tokens_b=None,
                            cls_token='<cls>', eos_token='<sep>'):
    '''
    如果只输入 tokens_a --> ['cls', tokens in a, 'sep'] 对应 [0, ..0.., 0],
    如果输入 tokens_a, tokens_b --> ['cls', tokens in a, 'sep', tokens in b, 'sep'] 对应 [0, ..0.., 0, ..1.., 1]
    '''
    tokens = [cls_token] + tokens_a + [eos_token]
    segments = [0] * len(tokens)

    if tokens_b is not None:
        tokens += tokens_b + [eos_token]
        segments += [1]*(len(tokens_b) + 1)

    # 拼接两个tokens list和<cls><sep>, 并输出它们的segments
    return tokens, segments












def _if_next_sentence(sentence, next_sentence, corpus):
    # corpus: lists of list of at least 2 sentences.
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(corpus)) # 随机选一个paragraph, 再从该段随机选一句
        is_next = False

    return sentence, next_sentence, is_next # 输出1个sentence, next sentence, 是否连接flag








# for nsp data whose datapoint is [token list, segments, nsp_label]
def _get_tokens_segments_nsplabels(corpus, max_len,
                                   cls_token='<cls>', eos_token='<sep>'):
    '''
    input:
        corpus: 3-D list of paragraphs. paragraph is a 2-D list of sentences. sentence is a 1-D list of tokens
        max_len
    
    explain:
        corpus: list of paragraph
        paragraph: list of sentence
        sentence: list of tokens

    return:
        list of [tokens, segments, is_next]
    '''
    nsp_data = []

    for paragraph in corpus:  # 取当前 paragraph: 2-D list, list of sentence, sentence is list of tokens

        for i in range(len(paragraph)-1):

            tokens_a, tokens_b, is_next = _if_next_sentence(paragraph[i], paragraph[i+1], corpus)
            # tokens_a和tokens_b要去掉可能存在于原文本的<cls><sep>
            tokens_a = [token for token in tokens_a if token not in (cls_token, eos_token)]
            tokens_b = [token for token in tokens_b if token not in (cls_token, eos_token)]

            if len(tokens_a) == 0 or len(tokens_b) == 0: # 如果任一句子去掉<cls>和<sep>后, 长度为0, 放弃
                continue
            if len(tokens_a) + len(tokens_b) + 3 > max_len: # 如果当前前后两句token数量+<cls>+2个<sep>数量超过max_len, 放弃
                continue # 即放弃拼接后过长的前后句对. 相当于truncate处理. 所以后面只需pad到这个max_len即可
            
            # list of tokens, list of 0-1
            tokens, segments = _concate_tokenlist_pair(tokens_a, tokens_b)

            nsp_data.append( [tokens, segments, is_next] )
    
    return nsp_data # list of [tokens, segments, is_next]





# for mlm data whose datapoint is [token list, mask_labels, mask positions]
def _mask_mapping_on_tokenList(tokens, vocab, donot_mask_tokens, mask_token='<mask>', mask_ratio=0.15):
    '''
    输入
        1. 原始 token list
        2. vocab: 词汇表
        3. 该token list要mask的token数量
    输出
        1. mask后 token list
        2. 被 mask 的原始 tokens
        3. 被 mask 的对应 positions
    '''
    candidate_mask_positions = [] # <cls> 和 <sep> 不被 mask
    for i, token in enumerate(tokens):
        if token not in donot_mask_tokens:
            candidate_mask_positions.append(i)
    
    # 可供mask的位置应该多于 1. 若小于等于1, 中断并打印错误信息
    assert len(candidate_mask_positions) > 1, f'invalid token list processing sentence {tokens}'

    num_mlm_masks = max(1, round( len(tokens)*mask_ratio ))

    mlm_tokens = copy.deepcopy(tokens) # 不能改变输入 tokens, 因为还要从中获取 true label tokens
    mask_positions_tokens = [] # 记录(mask_position, mask_tokens) pair

    random.shuffle(candidate_mask_positions) # 打乱

    for mask_position in candidate_mask_positions:
        if len(mask_positions_tokens) >= num_mlm_masks: # 当已经作了足够次数mask操作, 退出循环
            break

        if random.random() < 0.8: # 80%的概率, 用<mask>去mask
            mask = mask_token
        else:
            if random.random() < 0.5: # 10%的概率, 用随机token去mask
                mask = random.choice(vocab.tokens)

            else: # 10%的概率, 用自身去mask
                mask = tokens[mask_position]

        mlm_tokens[mask_position] = mask
        mask_positions_tokens.append( (mask_position, tokens[mask_position]) ) # 记录被mask的token位置, 以及真实token
    
    mask_positions_tokens = sorted(mask_positions_tokens, key=lambda x: x[0]) # 将 mask_positions_tokens 按照positions从小到大排列
    mask_positions = [v[0] for v in mask_positions_tokens]
    mlm_label_tokens = [v[1] for v in mask_positions_tokens]


    return vocab[mlm_tokens], vocab[mlm_label_tokens], mask_positions # token IDs, masked true label token IDs, 和对应的 mask positions







# toknIDs_segments_maskpositions_mlmlabels_nsplabels
# two_sentence_toknIDs_list, mask_position_list, mlm_label_toknIDs_list, two_sentence_segment_01_list, is_next_flag_TF)

# pad to 统一 two_sentence_toknIDs_list / two_sentence_segment_01_list 到 max_len, 用 valid lens 记录 valid area 信息
# pad to 统一 mask_position_list / mlm_label_toknIDs_list 到 max_len * 0.15, 用 _mlm_valid_lens 记录 valid area 信息

def _build_dataset(data, max_len, pad_tokenID, cls_tokenID, mask_ratio=0.15):
    '''
    input:
        data: list of 
            two_sentence_toknIDs_list, mask_position_list, mlm_label_toknIDs_list, two_sentence_segment_01_list, is_next_flag_TF
            作为单条样本
        
        max_len
    
    分别用 padTokn_ID / 0 pad to 统一 two_sentence_toknIDs_list / two_sentence_segment_01_list 到 max_len
        用 valid lens 记录 valid area 信息

    分别用 0 / clsTokn_ID pad to 统一 mask_position_list / mlm_label_toknIDs_list 到 max_len * 0.15
        用 mlm_valid_lens 记录 valid area 信息

    '''
    max_num_masks = max(1, round( max_len*mask_ratio ))

    tokenID_sample, segments_sample, valid_lens = [], [], []
    mask_positions_sample, mlm_valid_lens, mlm_labels_sample = [], [], []
    nsp_labels = []

    for toknIDs, mask_positions, mlmLabels, segments, nspLabels in data:
        # 取单条样本

        # 当前样本 sequence 的 valid length
        valid_lens.append( len(toknIDs) )

        # vocab<pad> pad tokens_idx 到 max_len
        tokenID_sample.append( truncate_pad(toknIDs, max_len, pad_tokenID) )

        # 0 pad segments 到 max_len
        segments_sample.append( truncate_pad(segments, max_len, 0) )

        # 当前 样本 mask_positions 的 valid length
        mlm_valid_lens.append( len(mask_positions) )

        # 0 pad mask_positions 到 max_num_masks
        mask_positions_sample.append( truncate_pad(mask_positions, max_num_masks, 0) )

        # 0 作为 position ID 代表的是 <cls>, 故对应应该用 clsTokn_ID 去pad mlm_labels. 不过其实这个不重要

        # clsTokn_ID pad 到 mlm_labels_idx
        mlm_labels_sample.append( truncate_pad(mlmLabels, max_num_masks, cls_tokenID) )

        nsp_labels.append( nspLabels )

    valid_lens = torch.tensor( valid_lens, dtype=torch.int32 ) # (sample_size,)
    tokenID_sample = torch.tensor( tokenID_sample, dtype=torch.int64 ) # (sample_size, max_len)
    segments_sample = torch.tensor( segments_sample, dtype=torch.int64 ) # (sample_size, max_len)

    mlm_valid_lens = torch.tensor( mlm_valid_lens, dtype=torch.int32 ) # (sample_size,)
    mask_positions_sample = torch.tensor( mask_positions_sample, dtype=torch.int64 ) # (sample_size, max_num_masks)
    mlm_labels_sample = torch.tensor( mlm_labels_sample, dtype=torch.int64 ) # (sample_size, max_num_masks)

    nsp_labels_sample = torch.tensor( nsp_labels, dtype=torch.int64 ) # (sample_size,)


    return tokenID_sample, valid_lens, segments_sample, mask_positions_sample, mlm_valid_lens, mlm_labels_sample, nsp_labels_sample




class wikitextDataset(torch.utils.data.Dataset):
    def __init__(self, fpath, vocab_path, max_len, cls_token='<cls>', eos_token='<sep>', mask_token='<mask>', mask_ratio=0.15):
        super().__init__()

        corpus = _read_shuffle(fpath) # 2-D list, list of list of at least 2 sentences(string)

        # 得到 vocab. 用 vocab.tokens 和 vocab.to_tokens(vocab.eow) 组建 glossary
        self._vocab = Vocab()
        self._vocab.load(vocab_path)

        glossary = {'tokens':self._vocab.tokens, "EOW_token":self._vocab.to_tokens(self._vocab.eow)}


        # corpus 是 3-D list: list of paragraph
        # paragraph 是 2-D list: list of sentence
        # sentence 是 1-D list: list of tokens
        tokens_corpus = _preprocess_tokenize(corpus, glossary, UNK_token='<unk>',
                                             need_lower=True, separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|')


        # tokens, 对应 segments, nsp labels
        tokens_segments_nsplabels = _get_tokens_segments_nsplabels(tokens_corpus, max_len, cls_token, eos_token)

        # 在 tokens 基础上, 得到 mlm 任务相关, 并映射 token 为 ID
        sample = []

        for tokens, segments, nsplabels in tokens_segments_nsplabels:
            # masked token_idx list, mask true token_idx list, mask position list
            toknIDs, mlmlabels, mask_positions = _mask_mapping_on_tokenList(tokens, self._vocab,
                                                                            donot_mask_tokens=(cls_token, eos_token),
                                                                            mask_token=mask_token, mask_ratio=mask_ratio)
            

            sample.append([toknIDs, segments, mask_positions, mlmlabels, nsplabels])
            # token ID list, segments 01 list, mask positions list, masked token ID list, is_next T/F
        
        self._sample_size = len(sample)

        # build dataset
        # _tokenID: (sample_size, max_len)
        # _valid_lens: (sample_size,)
        # _segment: (sample_size, max_len)
        # _mask_position: (sample_size, max_num_masks)
        # _mlm_valid_lens: (sample_size,)
        # _mlm_label: (sample_size, max_num_masks)
        # _nsp_label: (sample_size,)
        self._tokenID, self._valid_lens, self._segment, self._mask_position, self._mlm_valid_lens, self._mlm_label, self._nsp_label = \
            _build_dataset(data = sample, max_len = max_len, pad_tokenID = self._vocab['<pad>'],
                           cls_tokenID = self._vocab['<cls>'], mask_ratio=mask_ratio)
        


    def __getitem__(self, index):
        '''
        network input:
            tokens: (batch_size, seq_len)int64 ot token ID. 已包含<cls>和<sep>
            valid_lens: (batch_size,)
            segments: (batch_size, seq_len)01 分别代表 seq1 & seq2 | None, None 代表当前 batch 不需要进入 NSP task,
            mask_positions: (batch_size, num_masktks) | None, None 代表当前 batch 不需要进入 MLM task
        loss input:
            mlm_valid_lens: (batch_size,) 说明 mask position 是否是 pad. 如果是 pad 则为 invalid, 不计入 loss
            mlm_labels: (batch_size, num_masktks)int64 ot token ID.
            nsp_labels: (batch_size,)T/F
        '''
        return (
            tuple( [self._tokenID[index], self._valid_lens[index], self._segment[index], self._mask_position[index]] ),
            tuple( [self._mlm_valid_lens[index], self._mlm_label[index], self._nsp_label[index]] )
            )


    def __len__(self):
        return self._sample_size
    

    @property
    def vocab(self):
        return self._vocab
    