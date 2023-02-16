import torch
from torch import nn
from ...Compute.PredictTools import easyPredictor
from ...Compute.EvaluateTools import bleu
from .Dataset import build_tensorDataset
from ...Utils.Text.TextPreprocess import preprocess_space

def greedyPred_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    """
    序列到序列的greedy search预测
    src_sentence: 输入前需要lower/替换非正常空格为单空格/文字和,.?!之间需要有单空格
    num_steps: 预测步长. 预测出eos提前停止. 也是训练集中用到的num_steps(即seq_lens)

    因为训练集截断了num_steps之后的序列, 所以模型不能捕捉num_steps后面的序列关系.
    所以也需要对src_sentence加eos后作截断
    """
    net.eval()
    src_tokens = [src_sentence.split(' '), ] # 2D list
    enc_X, enc_valid_lens = build_tensorDataset(src_tokens, src_vocab, num_steps) # (1, num_steps), (1,)
    enc_X, enc_valid_lens = enc_X.to(device), enc_valid_lens.to(device)
    enc_outputs = net.encoder(enc_X, enc_valid_lens)
    enc_info = net.decoder.init_state(enc_outputs)
    infer_recorder = {}
    dec_X = torch.tensor( [tgt_vocab['<bos>'],], dtype=torch.int64, device=device).unsqueeze(0)
    output_idxs = []
    for _ in range(num_steps):
        Y, infer_recorder = net.decoder(dec_X, enc_info, infer_recorder)
        pred_idx = Y.argmax(dim=-1).item()
        if pred_idx == tgt_vocab['<eos>']:
            break
        if infer_recorder:
            output_idxs.append(pred_idx)
    if len(output_idxs) == 0:
        raise ValueError('Null sentence predicted')
    return ' '.join(tgt_vocab.to_tokens(output_idxs))

class sentenceTranslator(easyPredictor):
    def __init__(self, search_mode='greedy', bleu_k=2, device=None):
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.bleu_k = bleu_k
        if search_mode == 'greedy':
            self.pred_fn = greedyPred_seq2seq
        elif search_mode == 'beam':
            raise NotImplementedError
        else:
            raise ValueError('search_mode should be either "greedy" or "beam"')
        self.eval_fn = bleu

    def predict(self, src_sentence, net, src_vocab, tgt_vocab, num_steps):
        self.src_sentence = preprocess_space(src_sentence, need_lower=True, separate_puncs=',.!?')
        net.to(self.device)
        self.tgt_sentence = self.pred_fn(net, self.src_sentence, src_vocab, tgt_vocab, num_steps, self.device)
        return self.tgt_sentence

    def evaluate(self):
        assert hasattr(self, 'tgt_sentence'), 'pred target sentence not found'
        return self.eval_fn(self.tgt_sentence, self.src_sentence, self.bleu_k)
