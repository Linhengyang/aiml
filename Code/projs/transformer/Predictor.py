import torch
from torch import nn
import math
from ...Compute.PredictTools import easyPredictor
from ...Compute.EvaluateTools import bleu
from .Dataset import build_tensorDataset
from ...Utils.Text.TextPreprocess import preprocess_space

def str_to_enc_inputs(src_sentence, src_vocab, num_steps, device):
    '''
    src_sentence输入前需要预处理: lower/替换非正常空格为单空格/文字和,.?!之间需要有单空格
    因为训练集截断了num_steps之后的序列, 所以模型不能捕捉num_steps后面的序列关系.
    所以也需要对src_sentence作: 1预处理, 2加eos, 3长度控制在num_steps
    '''
    src_tokens = [src_sentence.split(' '), ] # 2D list, (1, seq_length)
    enc_X, enc_valid_lens = build_tensorDataset(src_tokens, src_vocab, num_steps)# shape (1, num_steps), (1,)
    return enc_X.to(device), enc_valid_lens.to(device)

def greedy_predict(net, tgt_vocab, num_steps, enc_inputs, device, alpha, *args):
    net.eval()
    enc_info = net.decoder.init_state(net.encoder(*enc_inputs))
    dec_X = torch.tensor( [tgt_vocab['<bos>'],], dtype=torch.int64, device=device).unsqueeze(0)
    output_idxs, infer_recorder, pred_score = [], {}, 0
    for i in range(num_steps):
        Y, infer_recorder = net.decoder(dec_X, enc_info, infer_recorder)
        assert infer_recorder['0'].size(1) == i+1, 'infer_recorder set wrong. please check infer code'
        pred_idx = Y.argmax(dim=-1).item()
        if pred_idx == tgt_vocab['<eos>']:
            break
        output_idxs.append(pred_idx)
        pred_score += torch.log( nn.Softmax(dim=-1)(Y).max(dim=-1).values ).item()
    long_award = math.pow(len(output_idxs), -alpha) if len(output_idxs) > 0 else 1
    return ' '.join(tgt_vocab.to_tokens(output_idxs)), pred_score * long_award

def beam_search(net, k_pred_token_mat, k_cond_prob_mat, enc_info, beam_size, vocab_size):
    # k_pred_token_mat: (beam_size, t-1)int64, k_cond_prob_mat: (beam_size, t-1)
    logits, _ = net.decoder(k_pred_token_mat, enc_info) # logits: (beam_size, t-1, vocab_size)
    cond_probs_t = nn.Softmax(dim=-1)( logits[:, -1, :] ) # (beam_size, vocab_size)
    cond_probs_seq = k_cond_prob_mat.prod(dim=1, keepdim=True) * cond_probs_t # (beam_size, vocab_size)
    topk = torch.topk(cond_probs_seq.flatten(), beam_size)
    row_indices = torch.div(topk.indices, vocab_size, rounding_mode='floor') # (beam_size, )
    col_indices = topk.indices % vocab_size # (beam_size, )
    k_cond_probs_t = cond_probs_t[row_indices, col_indices] # (beam_size, )
    next_pred_token_mat = torch.cat([k_pred_token_mat[row_indices], col_indices.reshape(-1,1)], dim=1) #(beam_size, t)
    next_cond_prob_mat = torch.cat([k_cond_prob_mat[row_indices], k_cond_probs_t.reshape(-1, 1)], dim=1) #(beam_size, t)
    return next_pred_token_mat.type(torch.int64), next_cond_prob_mat 

def mask_before_eos(pred_tokens_matrix, eos):
    bool_logits = pred_tokens_matrix != torch.tensor(eos)
    prob_mask = bool_logits.type(torch.int32).cumprod(dim=1)
    token_mask = (( prob_mask - torch.tensor(0.5) ) * 2 ).type(torch.int32)
    return token_mask, prob_mask

def beam_predict(net, tgt_vocab, num_steps, enc_inputs, device, alpha, beam_size):
    net.train()
    with torch.no_grad():
        src_enc_seqs, src_valid_lens = net.decoder.init_state(net.encoder(*enc_inputs))# (1, num_steps, d_dim), (1, )
        enc_info = (src_enc_seqs.repeat(beam_size, 1, 1), src_valid_lens.repeat(beam_size))
        dec_X = torch.tensor( [tgt_vocab['<bos>'],], dtype=torch.int64, device=device).unsqueeze(0)
        #first pred
        logits, _ = net.decoder(dec_X, (src_enc_seqs, src_valid_lens)) #logits shape(batch_size=1, num_steps=1, vocab_size)
        topk_1 = torch.topk(logits.flatten(), beam_size)
        pred_token_mat_1 = topk_1.indices.unsqueeze(1).type(torch.int64) #(beam_size, 1)
        cond_prob_mat_1 = nn.Softmax(dim=-1)(topk_1.values).unsqueeze(1) #(beam_size, 1)
        all_pred_token_mat, all_cond_prob_mat = [pred_token_mat_1], [cond_prob_mat_1]
        k_pred_token_mat, k_cond_prob_mat = pred_token_mat_1, cond_prob_mat_1
        for i in range(num_steps-1):
            k_pred_token_mat, k_cond_prob_mat = beam_search(net, k_pred_token_mat, k_cond_prob_mat, enc_info, beam_size, len(tgt_vocab))
            all_pred_token_mat.append(k_pred_token_mat)
            all_cond_prob_mat.append(k_cond_prob_mat)
        predseq_score_maps = {}
        for i in range(num_steps):
            k_pred_tokens = all_pred_token_mat[i] # k_pred_tokens shape: (k, i+1)
            related_cond_probs = all_cond_prob_mat[i] # related_cond_probs shape: (k, i+1)
            token_mask, prob_mask = mask_before_eos(k_pred_tokens, eos=tgt_vocab['<eos>'])
            k_pred_tokens = (k_pred_tokens * token_mask).type(torch.int64)
            log_cond_probs = (torch.log(related_cond_probs) * prob_mask)
            for j in range(beam_size):
                pred_seq = ' '.join(tgt_vocab.to_tokens([token for token in k_pred_tokens[j, :] if token >= 0]))
                valid_length = prob_mask[j, :].sum().item()
                if valid_length == 0:
                    continue
                predseq_score_maps[pred_seq] = log_cond_probs[j, :].sum().item() * math.pow(valid_length, -alpha)
    return max(predseq_score_maps, key= lambda x: predseq_score_maps[x]), predseq_score_maps

class sentenceTranslator(easyPredictor):
    def __init__(self, search_mode='greedy', bleu_k=2, device=None, beam_size=3, alpha=0.75):
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.bleu_k = bleu_k
        if search_mode == 'greedy':
            self.pred_fn = greedy_predict
            self.beam_size = None
            self.alpha = alpha # alpha越大, 输出越偏好长序列
        elif search_mode == 'beam':
            self.pred_fn = beam_predict
            self.beam_size = beam_size
            self.alpha = alpha # alpha越大, 输出越偏好长序列
        else:
            raise ValueError('search_mode should be either "greedy" or "beam"')
        self.eval_fn = bleu

    def predict(self, src_sentence, net, src_vocab, tgt_vocab, num_steps):
        self.src_sentence = preprocess_space(src_sentence, need_lower=True, separate_puncs=',.!?')
        enc_inputs = str_to_enc_inputs(self.src_sentence, src_vocab, num_steps, self.device)
        net.to(self.device)
        self.tgt_sentence, self._pred_scores = self.pred_fn(net, tgt_vocab, num_steps, enc_inputs, self.device, self.alpha, self.beam_size)
        return self.tgt_sentence

    def evaluate(self):
        assert hasattr(self, 'tgt_sentence'), 'pred target sentence not found'
        return self.eval_fn(self.tgt_sentence, self.src_sentence, self.bleu_k)

    @property
    def pred_scores(self):
        return self._pred_scores