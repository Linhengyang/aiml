import torch
from torch import nn
import math
from ...Compute.PredictTools import easyPredictor
from ...Compute.EvaluateTools import bleu
from .Dataset import build_tensorDataset
from ...Utils.Text.TextPreprocess import preprocess_space
import typing as t

def str_to_enc_inputs(src_sentence, src_vocab, num_steps, device):
    '''
    src_sentence输入前需要预处理: lower/替换非正常空格为单空格/文字和,.?!之间需要有单空格
    因为训练集截断了num_steps之后的序列, 所以模型不能捕捉num_steps后面的序列关系.
    所以也需要对src_sentence作: 1预处理, 2加eos, 3长度控制在num_steps
    '''
    src_tokens = [src_sentence.split(' '), ] # 2D list, (1, seq_length)
    src_array, src_valid_len = build_tensorDataset(src_tokens, src_vocab, num_steps)# shape (1, num_steps), (1,)

    return src_array.to(device), src_valid_len.to(device)



def greedy_predict(net, tgt_vocab, num_steps, enc_inputs, device, alpha, *args):
    net.eval()
    with torch.no_grad():
        # encoder / decoder.init_state in infer mode:
        # encoder input: enc_inputs = (src, src_valid_lens): shapes  [(1, num_stepss), (1,)]
        # init_state output: src_enc_info = (src_enc, src_valid_lens): [(1, num_stepss, d_dim), (1,)]
        src_enc_info = net.decoder.init_state(net.encoder(*enc_inputs))

        # decoder in infer mode:
        # input: tgt_query shape: (1, 1)int64, 对于第i次infer, tgt_query 的 timestep 是 i-1 (i = 1, 2, ..., num_steps), 前向的output的timestep 是 i
        #        src_enc_info = (src_enc, src_valid_lens): [(1, num_stepss, d_dim), (1,)]
        #        input KV_Caches: 
        #           Dict with keys: block_ind,
        #           values: 对于第 1 次infer, KV_Caches 为 空
        #                   对于第 i > 1 次infer, KV_Caches 是 tensors shape as (1, i-1, d_dim), i-1 维包含 timestep 0 到 i-2
        tgt_query = torch.tensor( [tgt_vocab['<bos>'],], dtype=torch.int64, device=device).unsqueeze(0)

        output_tokenIDs, KV_Caches, raw_pred_score = [], {}, 0

        for i in range(1, num_steps+1):
            # 对于第 i 次 infer: 
            # output: tgt_next_hat shape: (1, 1, vocab_size)tensor of logits, 对于第i次infer, timestep 是 i;
            #         output KV_Caches: dict of 
            #                      keys as block_indices
            #                      values as (1, i, d_dim) tensor, i 维 包含 timestep 0-i-1, 实际上就是 input KV_Caches 添加 timestep i-1 的 tgt_query
            Y_hat, KV_Caches = net.decoder(tgt_query, src_enc_info, KV_Caches)

            pred_tokenIDX = Y_hat.argmax(dim=-1).item()

            if pred_tokenIDX == tgt_vocab['<eos>']:
                break

            output_tokenIDs.append(pred_tokenIDX)
            raw_pred_score += torch.log( nn.Softmax(dim=-1)(Y_hat).max(dim=-1).values ).item() # raw_pred_score = log(estimated probability of selected one)
            
        long_award = math.pow(len(output_tokenIDs), -alpha) if len(output_tokenIDs) > 0 else 1
    
    return ' '.join(tgt_vocab.to_tokens(output_tokenIDs)), raw_pred_score * long_award



# greedy search 贪心搜索: 第 t(1=1,2,...num_steps)次infer, 从 Seq(0至t-1), net on Seq(0至t-1) 出 Tok(t) 的 vocab_size 个选择, 选择其中概率最大的作为 Tok(t)


# beam search 束搜索: 第 t(1=1,2,...num_steps)次infer, 不仅有 Seq(0至t-1), 还有 condProbs(0至t-1), condProbs的元素是 Tok(i) 在使用 net Seq(0至i-1) 生成时
# 的概率. 这里 condProbs(0) = 1. 
# 这个概率是条件概率: 在 net Tok(t) 时, net on Seq(0至t-1) 出 Tok(t) 的 vocab_size 个选择, softmax 后就是 vocab_size 个 Tok(t)|Seq(0至t-1) 的条件概率

# beam search 时, 每次选择都保留 k 个结果, 即有 Seq(0至t-1)_1, ..., Seq(0至t-1)_k, 和对应的 condProbs(0至t-1)_1,..., condProbs(0至t-1)_k

# 在寻找 Tok(t) 时, net on Seq(0至t-1)_1, ..., Seq(0至t-1)_k, 得到 k * vocab_size 个选择, 即一张 shape 为 (k, vocab_size) 的 token logits mat.
# token logits mat 的 第 j 行, 是 Seq(0至t-1)_j 作为前置序列, 生产出来的给 Tok(t) 的 vocab_size 个选择的 logits, softmax 后即为 vocab_size 个 Tok(t)|Seq(0至t-1)_j 的条件概率

# 求出这 k * vocab_size 个选择各自的条件概率, 即 softmax(dim=-1) on (k, vocab_size) 的 token logits mat --> token condProb mat, shape 为 (k, vocab_size)

# 那么 Seq(0至t-1)_j + Tok(t) 的序列总概率, 即为 Tok(t)|Seq(0至t-1)_j 的条件概率 * 连乘 condProbs(0至t-1)_j. 对于 Seq(0至t-1)_j 为前置序列, 有 vocab_size 个序列总概率
# 对于 k 条 前置序列 Seq(0至t-1)_1, ..., Seq(0至t-1)_k, 得到 k * vocab_size 个序列总概率, 
# 是一张 shape 为 (k, vocab_size) 的矩阵 seq prob mat: 元素为 序列总概率, 行坐标代表前置序列, 列坐标代表 Tok(t)
# 取 seq prob mat 的 前 k 个最大值, top1, top2,...topk. 对于 j = 1,2..k, topj 的行坐标代表的前置序列, append topj 的列坐标代表的 Tok(t), 得到了 序列总概率最大的
# k 个 Seq(0至t), 分别是 Seq(0至t)_1,...Seq(0至t)_k

# 对于第 t 步predict
def beam_search_single_step(k, k_seq_mat, k_cond_prob_mat, net, vocab_size, src_enc_info, parrallel=False, k_KV_Caches=None):
    # beam_size == k
    # k_seq_mat: (k, t)int64, 包含 timestep 0 至 t-1, k 条 Seq(0至t-1), 每条作为行. 对于第1步predict, k_seq_mat 是 k 条  timestep=0的<bos>
    # k_cond_prob_mat: (k, t), 包含 timestep 0 至 t-1, k 条 Cond Prob序列(0至t-1), 每条作为行. timestep=0, Cond Prob=1
    # KV_Caches: dict, 对于 第 1 次predict, 为空 {}
    #   对于第 t > 1 次predict, keys 是 block_inds, values 是 tensors shape as (1, t-1, d_dim), i-1 维包含 timestep 0 到 i-2

    # cond_prob_tokn_t_mat = torch.rand((k, vocab_size)) # (k, vocab_size), TODO

    if parrallel and net.training:
        logits, _ = net.decoder(k_seq_mat, src_enc_info) # (k, num_steps, vocab_size)tensor of logits,  timestep 从 1 到 t;
        logits_tokn_t = logits[:, -1, :].squeeze(1) # (k, 1, vocab_size) --> (k, vocab_size)
        cond_prob_tokn_t_mat = nn.Softmax(dim=-1)(logits_tokn_t) # (k, vocab_size)
    elif isinstance(KV_Caches, dict):


    prob_seqs = k_cond_prob_mat.prod(dim=1, keepdim=True) * cond_prob_tokn_t_mat # (k, vocab_size)

    topk = torch.topk(prob_seqs.flatten(), k)

    row_inds = torch.div(topk.indices, vocab_size, rounding_mode='floor') # (k, )
    col_inds = topk.indices % vocab_size # (k, )

    k_cond_prob_tokn_t = cond_prob_tokn_t_mat[row_inds, col_inds] # (k, ), 从 timestep t 的 cond_prob_tokn_t_mat (k, vocab_size) 中选出的 top k的 条件概率

    # append selected top k's 条件概率 to k_cond_prob_mat
    next_k_cond_prob_mat = torch.cat([k_cond_prob_mat[row_inds], k_cond_prob_tokn_t.reshape(-1, 1)], dim=1) #(beam_size, t)

    # append selected top k's token ID to k seqs
    next_k_seq_mat = torch.cat([k_seq_mat[row_inds], col_inds.reshape(-1,1)], dim=1) #(k, t+1)

    return next_k_seq_mat.type(torch.int64), next_k_cond_prob_mat 



def mask_before_eos(pred_tokens_matrix, eos):
    bool_logits = pred_tokens_matrix != torch.tensor(eos)
    prob_mask = bool_logits.type(torch.int32).cumprod(dim=1)
    token_mask = (( prob_mask - torch.tensor(0.5) ) * 2 ).type(torch.int32)

    return token_mask, prob_mask



def beam_predict(net, tgt_vocab, num_steps, enc_inputs, device, alpha, beam_size):
    net.train() #in beam predict, need to predict on beam_size seqs simultaneously, so train mode is used
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
        super().__init__()
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.bleu_k = bleu_k
        if search_mode == 'greedy':
            self.pred_fn = greedy_predict
            self.beam_size = None
        elif search_mode == 'beam':
            self.pred_fn = beam_predict
            self.beam_size = beam_size
        else:
            raise NotImplementedError(
                f'search_mode {search_mode} not implemented. Should be one of "greedy" or "beam"'
                )
        
        self.alpha = alpha # alpha越大, 输出越偏好长序列
        self.eval_fn = bleu

    def predict(self, src_sentence, net, src_vocab, tgt_vocab, num_steps):

        # 预处理: 小写化, 替换不间断空格为单空格, 并trim首尾空格, 保证文字和,.!?符号之间有 单空格. normalize 空白字符
        self.src_sentence = preprocess_space(src_sentence, need_lower=True, separate_puncs=',.!?', normalize_whitespace=True)

        enc_inputs = str_to_enc_inputs(self.src_sentence, src_vocab, num_steps, self.device)
        net.to(self.device)
        self.pred_sentence, self._pred_scores = self.pred_fn(net, tgt_vocab, num_steps, enc_inputs, self.device, self.alpha, self.beam_size)

        return self.pred_sentence

    def evaluate(self, tgt_sentence):
        assert hasattr(self, 'pred_sentence'), 'predicted target sentence not found'
        self.tgt_sentence = preprocess_space(tgt_sentence, need_lower=True, separate_puncs=',.!?')
        return self.eval_fn(self.pred_sentence, self.tgt_sentence, self.bleu_k)

    @property
    def pred_scores(self):
        return self._pred_scores