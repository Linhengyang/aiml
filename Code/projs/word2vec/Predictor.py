import torch
from torch import nn
import math
from ...Compute.PredictTools import easyPredictor
from .Network import skipgramNegSp, cbowNegSp

def get_k_synonyms(query_token, k, vocab, net):
    if isinstance(net, skipgramNegSp): # 对于skipgram模型, 用它的center词embed作为表示
        embed = net.center_embed
    elif isinstance(net, cbowNegSp): # 对于cbow模型, 用它的context词embed作为表示
        embed = net.context_embed
    else:
        raise ValueError('input net must be one of skip-gram/cbow')
    W = embed.weight.data #(vocab_size, embed_size)
    x = W[vocab[query_token]] #(embed_size, ), query词的embed表示
    cosine = torch.mv(W, x)/torch.sqrt(W.pow(2).sum(dim=1) * x.pow(2).sum() + 1e-9)
    topk = torch.topk(cosine, k=k+1)
    synonyms = topk.indices[1:].cpu().numpy().astype('int32').tolist()
    similarity = topk.values[1:].cpu().numpy().tolist()
    return vocab.to_tokens(synonyms), similarity

class wordInference(easyPredictor):
    def __init__(self, device=None):
        super().__init__()
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.pred_fn = get_k_synonyms

    def predict(self, query_token, k, vocab, net):
        net.to(self.device)
        self.preds, self._pred_scores = self.pred_fn(query_token, k, vocab, net)
        return self.preds

    def evaluate(self):
        pass

    @property
    def pred_scores(self):
        return self._pred_scores
