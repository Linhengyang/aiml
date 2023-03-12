import torch
from torch import nn
import math
from ...Compute.PredictTools import easyPredictor
from ...projs.bert.Dataset import get_tokens_segments
from ...Utils.Common.SeqOperation import truncate_pad

def get_bert_embedding(net, vocab, tokens_a, tokens_b, max_len, device):
    tokens, segments = get_tokens_segments(tokens_a, tokens_b)
    valid_lens = torch.tensor(len(tokens), dtype=torch.float32).unsqueeze(0) #(1,)
    tokens = truncate_pad(tokens, max_len, '<pad>')
    segments = truncate_pad(segments, max_len, 0)
    tokens_idx = torch.tensor(vocab[tokens], dtype=torch.int64, device=device).unsqueeze(0) #(1, seq_len)
    segments = torch.tensor(segments, dtype=torch.int64, device=device).unsqueeze(0) #(1, seq_len)
    embd_ = net(tokens_idx, segments, valid_lens)[0].squeeze(0) #(seq_len, num_hiddens)
    return embd_

class tokensEncoder(easyPredictor):
    def __init__(self, max_len, device=None):
        super().__init__()
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.max_len = max_len
        self.pred_fn = get_bert_embedding

    def predict(self, net, vocab, tokens_a, tokens_b=None):
        return self.pred_fn(net, vocab, tokens_a, tokens_b, self.max_len, self.device)

    def evaluate(self):
        pass

    @property
    def pred_scores(self):
        pass
