import torch.nn as nn
import math
import torch
from ...Modules._transformer import TransformerEncoderBlock
from ...Base.RootLayers.PositionalEncodings import LearnAbsPosEnc

class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, seq_len, use_bias=True, **kwargs):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = LearnAbsPosEnc(seq_len, num_hiddens, dropout) # seq_len是数据制作时, 输入batch align后的序列长度(text/text pair)
        self.seg_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            cur_blk = TransformerEncoderBlock(num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias)
            self.blks.add_module(f'blk{i+1}', cur_blk)
    
    def forward(self, tokens, segments, valid_lens):# bert的输入同样用pad补齐/截断, 所以不同batch的seq_len都是一样的
        # tokens shape: (batch_size, seq_len<=max_len)int64
        # segments shape: (batch_size, seq_len<=max_len)01 int64
        X = self.token_embedding(tokens) + self.seg_embedding(segments)
        X = self.pos_encoding(X)
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X # 输出 shape: (batch_size, seq_len, num_hiddens)

class MLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens):
        super().__init__()
        self.mlp = nn.Sequential(nn.LazyLinear(num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.LazyLinear(vocab_size))
    
    def forward(self, token_embd, pred_positions):
        # token_embd shape: (batch_size, seq_len, num_hiddens)
        # pred_positions: (batch_size, num_masktks)
        num_masktks = pred_positions.shape[1]
        pred_positions_flat = pred_positions.flatten() # (pos1_1, pos2_1, ...,pos_num_masktks_1, pos1_2, ...)
        batch_size = token_embd.shape[0]
        batch_idx = torch.arange(0, batch_size, device=token_embd.device) # (0 -> batch_size-1)
        # batch_idx = torch.repeat_interleave(batch_idx, num_masktks) # (0,0..0, 1,1,..1, .., bs-1, bs-1,...,bs-1)各num_masktks个
        batch_idx = batch_idx.repeat_interleave(num_masktks) # (0,0..0, 1,1,..1, .., bs-1, bs-1,...,bs-1)各num_masktks个
        mask_tokens_embd = token_embd[batch_idx, pred_positions_flat] # (batch_size * num_masktks, num_hiddens)
        mask_tokens_embd = mask_tokens_embd.reshape(batch_size, num_masktks, -1)  # (batch_size, num_masktks, num_hiddens)
        mask_Y_hat = self.mlp(mask_tokens_embd) # (batch_size, num_masktks, vocab_size)
        return mask_Y_hat # ground truth Y (batch_size, num_masktks), before go into CELoss, need to reshape mask_Y_hat

class NSP(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.LazyLinear(2)
    
    def forward(self, cls_X):
        # input cls_X shape: (batch_size, num_hiddens)
        return self.outputs(cls_X) # output shape: (batch_size, 2)

class BERT(nn.Module):
    def __init__(self, vocab_size, num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, seq_len):
        super().__init__()
        self.encoder = BERTEncoder(vocab_size, num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, seq_len)
        self.mlm = MLM(vocab_size, num_hiddens)
        self.hidden = nn.Sequential(nn.LazyLinear(num_hiddens),
                                     nn.Tanh())
        self.nsp = NSP()
    
    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        embd_X = self.encoder(tokens, segments, valid_lens) # (batch_size, seq_len, num_hiddens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(embd_X, pred_positions)# (batch_size, num_masktks, vocab_size)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(self.hidden(embd_X[:, 0, :])) # (batch_size, 2)
        return embd_X, mlm_Y_hat, nsp_Y_hat