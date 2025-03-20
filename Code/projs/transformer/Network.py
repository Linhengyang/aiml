from ...Base.MetaFrames import Encoder, Decoder, AttentionDecoder, EncoderDecoder
from ...Base.RootLayers.PositionalEncodings import TrigonoAbsPosEnc, LearnAbsPosEnc
from ...Modules._transformer import TransformerEncoderBlock, TransformerDecoderBlock
import torch.nn as nn
import math





class TransformerEncoder(Encoder):
    '''
    Transformer的 Encoder 部分由以下组成. 各层的 shape 变化如下:
    1. Embedding层. 输入(batch_size, seq_length), 每个元素是 0-vocab_size 的integer, 代表token ID。输出 (batch_size, seq_length, num_hiddens)
        Embedding层相当于一个 onehot + linear-projection 的组合体,
        (batch_size, seq_length) --onehot--> (batch_size, seq_length, vocab_size) --linear_proj--> (batch_size, seq_length, num_hiddens)
    2. PositionEncoding层. 输入 (batch_size, seq_length, num_hiddens). 输出 add position info 后的 (batch_size, seq_length, num_hiddens)
    3. 连续的 Encoder Block. 每个 EncoderBlock 的输入 src_X (batch_size, seq_length, num_hiddens), 输出 (batch_size, seq_length, num_hiddens)
        输入 valid_lens (batch_size,)
    
    在Encoder内部, 前后关系依赖是输入 timestep 1-seq_length, 输出 timestep 1-seq_length，作为对 input data 的深度表征
    '''
    def __init__(self, vocab_size, num_blk, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = TrigonoAbsPosEnc(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blk):
            cur_blk = TransformerEncoderBlock(num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias)
            self.blks.add_module("encblock"+str(i), cur_blk)

    def forward(self, src, src_valid_lens):
        # src shape: (batch_size, num_steps)int64, timestep: 1 -> num_steps
        # src_valid_lens shape: (batch_size,)int32
        src_embd = self.embedding(src)
        src_enc = self.pos_encoding(src_embd * math.sqrt(self.num_hiddens)) # 在embed后, 位置编码前, 将embed结果scale sqrt(d)

        for blk in self.blks:
            src_enc = blk(src_enc, src_valid_lens)
        
        return src_enc, src_valid_lens






class TransformerDecoder(AttentionDecoder):
    '''
    Transformer的 Decoder 部分由以下组成. 各层的 shape 变化如下:
    1. Embedding层. 输入(batch_size, seq_length), 每个元素是 0-vocab_size 的integer, 代表token ID。 输出(batch_size, seq_length, num_hiddens)
    2. pos_encoding层. 输入(batch_size, seq_length, num_hiddens), 输出(batch_size, seq_length, num_hiddens)

    '''
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
        # encoder returns encoded_src, src_valid_lens

        # train: src_enc_info = (src_enc, src_valid_lens): [(batch_size, num_steps, d_dim), (batch_size,)]
        # infer: src_enc_info = (src_enc, src_valid_lens): [(1, num_stepss, d_dim), (1,)]
        src_enc_info = enc_outputs

        return src_enc_info


    def forward(self, tgt_query, src_enc_info, KV_Caches=None):
        # train: tgt_query shape: (batch_size, num_steps)int64, timestep 从 0 到 num_steps-1
        #        src_enc_info = (src_enc, src_valid_lens): [(batch_size, num_steps, d_dim), (batch_size,)]
        #        KV_Caches: None

        # infer: tgt_query shape: (1, 1)int64, 对于第i次infer, tgt_query 的 timestep 是 i-1 (i = 1, 2, ..., num_steps), 前向的output的timestep 是 i
        #        src_enc_info = (src_enc, src_valid_lens): [(1, num_stepss, d_dim), (1,)]
        #        input KV_Caches: 
        #           Dict with keys: block_ind,
        #           values: 对于第 1 次infer, KV_Caches 为 空
        #                   对于第 i > 1 次infer, KV_Caches 是 tensors shape as (1, i-1, d_dim), i-1 维包含 timestep 0 到 i-2

        tgt_embd = self.embedding(tgt_query)
        tgt_query = self.pos_encoding(tgt_embd * math.sqrt(self.num_hiddens))

        # Decoder Block 的输入 tgt_query, src_enc_info, KV_Caches
        for blk in self.blks:
            tgt_query, KV_Caches = blk(tgt_query, src_enc_info, KV_Caches)
        
        #train: output[0] shape: (batch_size, num_steps, vocab_size)tensor of logits,  timestep 从 1 到 num_steps;
        #       output[1]: None
        #infer: output[0] shape: (1, 1, vocab_size)tensor of logits, 对于第i次infer, timestep 是 i;
        #       output[1]: dict of 
        #                      keys as block_indices
        #                      values as (1, i, d_dim) tensor, i 维 包含 timestep 0-i-1, 实际上就是 input KV_Caches 添加 timestep i-1 的 tgt_query
        return self.dense(tgt_query), KV_Caches







class Transformer(EncoderDecoder):
    ## 整体两种实现模式:
    # 1是encoder和decoder一起训练, 但分开使用, 这样transformer本身只需考虑train batch input的场景, infer的循环放在pred函数里
    # 2是encoder和decoder一起训练, 一起使用, 这样transformer要注意infer sample input的场景, infer的循环放在forward里. 即针对 net.training 的状态, 写不同的infer
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    # net_inputs_batch = (X_batch, Y_frontshift1_batch, X_valid_lens_batch),
    def forward(self, src, tgt_frontshift1, src_valid_lens):
        enc_outputs = self.encoder(src, src_valid_lens) # src_enc, src_valid_lens
        enc_info = self.decoder.init_state(enc_outputs)

        #train: output[0] shape: (batch_size, num_steps, vocab_size) tensor of logits, output[2]: None
        #infer: output[0] shape: (1, 1, vocab_size) tensor of logits, output[2]: dict of (1, cur_infer_step i, d_dim) tensor
        return self.decoder(tgt_frontshift1, enc_info)
