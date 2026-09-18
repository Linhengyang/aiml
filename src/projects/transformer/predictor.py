import torch
from torch import nn
import math
import re
from operator import itemgetter
import typing as t
from src.core.interface.infra_easy import easyPredictor
from src.core.evaluation.evaluate import bleu
from .dataset import tensorize_tokens, segment_seq2seqText, preprocess_space


class sentenceTranslator(easyPredictor):
    def __init__(self,
                 net, # net
                 vocab, max_gen_size, temporature, topk, # gen
                 bleu_k=2, device=None, length_factor=0.75, # performance
                 ):
        super().__init__()
        # load src/tgt language vocab
        self._vocab = vocab

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

    def predict(self,
                src_sentences: t.List[str],
                need_lower=True, 
                separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
        # 凑批
        sample_separator, srctgt_separator = '\n', '\t'

        raw_text = sample_separator.join([src + srctgt_separator + " " for src in src_sentences])
        source, _, _ = segment_seq2seqText(raw_text, sample_separator, srctgt_separator, self._vocab.to_glossary(),
                                           self._vocab.to_subwords(self._vocab.unk), need_lower, separate_puncs)
        src, src_valid_lens = tensorize_tokens(source, self._vocab, self.max_gen_size) #[B, max_gen_size=context_size], [B,]

        # output_tokens: [B, gen_size]
        output_ids = self.net.generate(src, src_valid_lens, self.max_gen_size, self._vocab['<bos>'], self._vocab['<eos>'], self.temporature, self.topk)
        sentences = self._vocab.to_subwords(output_ids.tolist()) # map to list of list of subwords
        eow = self._vocab.to_subwords(self._vocab.eow)
        if eow: # 如果 eow 不是空字符
            self.pred_sentences = [re.sub(eow, ' ', ''.join(subwords)) for subwords in sentences]
        else: # 如果 eow 是空字符
            self.pred_sentences = [' '.join(subwords) for subwords in sentences]
        
        return self.pred_sentences

    def evaluate(self, tgt_sentences, need_lower=True, separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
        assert hasattr(self, 'pred_sentences'), f'predicted target sentence not found'
        self.tgt_sentences = [preprocess_space(sentence, need_lower, separate_puncs, normalize_whitespace=True) for sentence in tgt_sentences]
        
        return [self.eval_fn(pred, tgt, self.bleu_k) for pred, tgt in zip(self.pred_sentences, self.tgt_sentences)]