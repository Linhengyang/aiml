from ...Base.MetaFrames import Encoder, Decoder, AttentionDecoder, EncoderDecoder
from ...Base.RootLayers.PositionalEncodings import TrigonoAbsPosEnc, LearnAbsPosEnc
from ...Modules._transformer import TransformerEncoderBlock, TransformerDecoderBlock
import torch.nn as nn
import math

class TransformerEncoder(Encoder):
    def __init__(self, vocab_size, num_blk, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = TrigonoAbsPosEnc(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blk):
            cur_blk = TransformerEncoderBlock(num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias)
            self.blks.add_module("encblock"+str(i), cur_blk)

    def forward(self, src_X, valid_lens):
        # src_X shape: (batch_size, num_steps), valid_lens shape: (batch_size,)
        embed_X = self.embedding(src_X)
        X = self.pos_encoding(embed_X * math.sqrt(self.num_hiddens)) # 在embed后, 位置编码前, 将embed结果scale sqrt(d)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
        return X, valid_lens

class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, num_blk, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = TrigonoAbsPosEnc(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blk):
            cur_blk = TransformerDecoderBlock(i, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias)
            self.blks.add_module("decblock"+str(i), cur_blk)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs):
        return enc_outputs # encoder returns encoded_src, src_valid_lens

    def forward(self, tgt_X, enc_info, infer_recorder=None):
        #train: tgt_X shape: (batch_size, num_steps), enc_info: [(batch_size, num_steps, d_dim), (batch_size,)]
        #infer: tgt_X shape: (1, 1), enc_info: [(1, num_stepss, d_dim), (1,)], A dict as infer_recorder
        emb_X = self.embedding(tgt_X)
        X = self.pos_encoding(emb_X * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            X, infer_recorder = blk(X, enc_info, infer_recorder)
        #train: output[0] shape: (batch_size, num_steps, vocab_size), output[2]: None
        #infer: output[0] shape: (1, 1, vocab_size), output[2]: dict of (1, cur_infer_step, d_dim) tensor
        return self.dense(X), infer_recorder

class Transformer(EncoderDecoder):
    ## 整体两种实现模式:
    # 1是encoder和decoder一起训练, 但分开使用, 这样transformer本身只需考虑train batch input的场景, infer的循环放在pred函数里
    # 2是encoder和decoder一起训练, 一起使用, 这样transformer要注意infer sample input的场景, infer的循环放在forward里
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src_X, tgt_X, src_valid_lens):
        enc_outputs = self.encoder(src_X, src_valid_lens)
        enc_info = self.decoder.init_state(enc_outputs)
        return self.decoder(tgt_X, enc_info) # 第1种实现, 在这里只考虑train mode. infer的时候把transformer拆开使用
