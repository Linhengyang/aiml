import torch
from torch import nn
import math
import re
from operator import itemgetter
import typing as t
from src.core.interface.infra_easy import easyPredictor
from src.core.evaluation.evaluate import bleu
from src.utils.text.text_preprocess import preprocess_space
from src.utils.text.string_segment import sentence_segment_greedy
from src.core.data.assemble import truncate_pad


class sentenceTranslator(easyPredictor):
    def __init__(self,
                 net, # net
                 vocab, max_gen_size, temporature, topk, # gen
                 bleu_k=2, device=None, length_factor=0.75, # performance
                 ):
        super().__init__()
        # load src/tgt language vocab
        self._vocab = vocab, 

        # set device
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            self.device = torch.device('cpu')
        print(f"use device {self.device} to infer")

        self.net = net
        self.max_gen_size = max_gen_size
        self.temporature = temporature
        self.topk = topk if isinstance(topk, int) else None

        # set evaluate function
        self.eval_fn = bleu
        self.bleu_k = bleu_k
        self.length_factor = length_factor # length_factor 越大, 输出越偏好长序列

    def predict(self, src_sentence, need_lower=True, separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
        self.src_sentence = preprocess_space(src_sentence, need_lower, separate_puncs, normalize_whitespace=True)
        source, _ = sentence_segment_greedy(
            self.src_sentence,
            self._vocab.to_glossary(),
            UNK_token = self._vocab.to_subwords(self._vocab.unk),
            flatten = True,
            need_preprocess = False # 已经有预处理了
            )

        # 和训练相同的tokenize: output src_array:(1, context_size)int64, valid_lens:(1,)int32
        src = self._vocab[ source ] + [ self._vocab['<eos>'] ]
        src = truncate_pad(source, self.net.decoder.decoder_context_size, self._vocab['<pad>'])
        src_ids = torch.tensor( src, dtype=torch.int64).unsqueeze(0) # (context_size,) --> (1, context_size)int64
        src_valid_lens = (src_ids != self._vocab['<pad>']).type(torch.int64).sum(1) # (1,)int64
        output_tokens = self.net.generate(src_ids, src_valid_lens, self.max_gen_size, self._vocab['<bos>'], self._vocab['<eos>'], self.temporature, self.topk)
        
        eow = self._vocab.to_subwords(self._vocab.eow)
        if eow: # 如果 eow 不是空字符
            self.pred_sentence = re.sub(eow, ' ', ''.join(output_tokens))
        else: # 如果 eow 是空字符
            self.pred_sentence = ' '.join(output_tokens)
        
        return self.pred_sentence

    def evaluate(self, tgt_sentence, need_lower=True, separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
        assert hasattr(self, 'pred_sentence'), f'predicted target sentence not found'
        self.tgt_sentence = preprocess_space(tgt_sentence, need_lower, separate_puncs, normalize_whitespace=True)
        
        return self.eval_fn(self.pred_sentence, self.tgt_sentence, self.bleu_k)