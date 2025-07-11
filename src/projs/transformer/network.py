from ...core.nn_components.meta_frames import Encoder, AttentionDecoder, EncoderDecoder
from ...core.nn_components.root_layers.positional_encodings import TrigonoAbsPosEnc
from ...core.nn_components.sub_modules._transformer import TransformerEncoderBlock, TransformerDecoderBlock
import torch.nn as nn
import math
import torch




class TransformerEncoder(Encoder):
    '''
    在Encoder内部, 前后关系依赖是输入 timestep 1-seq_length, 输出 timestep 1-seq_length, 实现对 input data 的深度表征

    1. Embedding层. 
        输入(batch_size, seq_length), 每个元素是 0-vocab_size 的integer, 代表token ID。输出 (batch_size, seq_length, num_hiddens)
        Embedding层相当于一个 onehot + linear-projection 的组合体,
        (batch_size, seq_length) --onehot--> (batch_size, seq_length, vocab_size) --linear_proj--> (batch_size, seq_length, num_hiddens)

    2. PositionEncoding层.
        输入 (seq_length,) 的 位置信息, 对其编码. 注意 encoder 里的位置信息是 1-seq_length, timestep 0 是 BOS, 不在 src seq里
        输出 (1, seq_length, num_hiddens)

    3. 连续的 Encoder Block.
        每个 EncoderBlock 的输入 src_embd + pos_embd (batch_size, seq_length, num_hiddens), 输出 (batch_size, seq_length, num_hiddens)
        输入/输出 valid_lens (batch_size,) 作为在 Block间不会变的 mask 信息接力传递. valid_lens[i] 给出了样本i 的 seq_length 中, 有几个是valid.

        在自注意力中, 对每个token(对应每条query)而言, 都只限制了整体valid 长度. 故 encoder 里, 每个token是跟序列里前/后所有valid token作相关性运算的
    '''
    def __init__(self, vocab_size, num_blk, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = TrigonoAbsPosEnc(num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_blk):
            cur_blk = TransformerEncoderBlock(num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias)
            self.blks.add_module("encblock"+str(i), cur_blk)

    def forward(self, src, src_valid_lens):
        # src shape: (batch_size, num_steps)int64, timestep: 1 -> num_steps
        _, num_steps = src.shape

        # src_valid_lens shape: (batch_size,)int32

        # source input data embedding
        # src_embd: shape (batch_size, num_steps, num_hiddens)
        src_embd = self.embedding(src) * math.sqrt(self.num_hiddens)
        # 使用 固定位置编码时, 避免位置编码的影响过大，所以放大input embeddings

        # source position embedding: 没有 bos, 从 1 开始到 num_steps
        # pos_embd: (num_steps,) --pos embed--> (1, num_steps, num_hiddens)
        # 1 到 num_steps, 所有位置都需要position embed. 0 是给 <bos> 的. src里没有bos
        position_ids = torch.arange(1, num_steps+1, dtype=torch.int64, device=src.device)

        # input embeddings + position embedding
        src_enc = self.dropout(src_embd + self.pos_encoding(position_ids))

        for blk in self.blks:
            src_enc = blk(src_enc, src_valid_lens)
        
        return src_enc, src_valid_lens






class TransformerDecoder(AttentionDecoder):
    '''
    train 模式:
    单次forward是 seq_length 并行, 前后关系依赖是输入 timestep 0-seq_length-1, 输出 timestep 1-seq_length, 实现对 shift1 data 的 并行预测

    1. Embedding层.
        输入(batch_size, seq_length), 每个元素是 0-vocab_size 的int64, 代表token ID. 输出(batch_size, seq_length, num_hiddens)

    2. pos_encoding层.
        输入 (seq_length,) 的 位置信息, 对其编码. 注意 decoder 里的位置信息是 0-seq_length-1, timestep 0 是 BOS, 在 tgt seq里
        输出 (1, seq_length, num_hiddens)

    3. 连续的 decoder Block.
        每个 DecoderBlock 的输入 tgt_embd + pos_embd (batch_size, seq_length, num_hiddens),
        输入/输出 src_enc_info(src_enc, src_valid_lens): [(batch_size, num_steps, d_dim), (batch_size,)] 作为在Block间不会变的 src 信息接力传递.
        输出 (batch_size, seq_length, num_hiddens)

        在 Block 内部, tgt_embd 先作 自回归的自注意力 以深度表征, 再和 src_embd 作 交叉注意力 以获取信息

        Block 之间没有传递 valid_lens of tgt seq 的信息. 这个 valid lens of tgt seq 用在了 求loss 的步骤里

        
    eval 模式:
    单次forward是生产 单个token 的过程. 总共要生成 seq_length 个token, 所以总过程要执行forward seq_length次,
    第 i 次 forward 生成 timestep 为 i 的token, i = 1,2,...,seq_length.    timestep = 0 的token是<BOS>

    对于 第 i 次forward, i = 1,2,...,seq_length
    1. Embedding层.
        输入(1, 1), 元素是 0-vocab_size 的int64, 代表 timestep=i-1 的 token ID. 输出 (1, 1, num_hiddens)
    
    2. pos_encoding层.
        对 (1,) 的 位置信息作编码. 这里这个位置信息代表 timestep i-1.
        在forward过程中, 依靠 KV_Caches 中, dim 1 的维度长度, 得知当前 tgt_query 的位置信息
        输出 (1, 1, num_hiddens)

    3. 连续的 decoder Block
        每个 DecoderBlock 的输入 tgt_embd + pos_embd (1, 1, num_hiddens)
        输入/输出 src_enc_info(src_enc, src_valid_lens): [(batch_size, num_steps, d_dim), (batch_size,)] 作为在Block间不会变的 src 信息接力传递.
        输入/输出 KV_Caches: 
            输入的 KV_Caches 记录了每个 Block 各自的 输入 tgt_tensor(timestep i-1) 在 timesteps 0 - i-2 上的堆叠,
            输出的 KV_Caches 记录了每个 Block 堆叠了 输入 tgt_tensor(timestep i-1) 更新后的结果. 一次forward过程中, 所有KV都更新一次
    
    4. dense层.
        输出 (1, 1, vocab_size)tensor of logits, 即对 timestep=i 的token 的预测
        输出 KV_Caches: 记录了每个 Block 各自的 输入 tgt_tensor 在 timesteps 0 - i-1 上的堆叠.
    '''
    def __init__(self, vocab_size, num_blk, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = TrigonoAbsPosEnc(num_hiddens)
        self.dropout = nn.Dropout(dropout)
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


    def forward(self, tgt_dec_input, src_enc_info, KV_Caches=None):
        # train: tgt_dec_input shape: (batch_size, num_steps)int64, timestep 从 0 到 num_steps-1
        #        src_enc_info = (src_enc, src_valid_lens): [(batch_size, num_steps, d_dim), (batch_size,)]
        #        KV_Caches: None

        #        position_ids: 训练阶段时, position_ids 应该是 tensor([0, 1, ..., num_steps-1])

        # 对于第i次infer: i = 1, 2, ..., num_steps
        #        tgt_dec_input shape: (1, 1)int64, 其 timestep 是 i-1, 前向的output的timestep 是 i
        #        src_enc_info = (src_enc, src_valid_lens): [(1, num_steps, d_dim), (1,)]
        #        input KV_Caches: 
        #           Dict with keys: block_ind,
        #           values: 对于第 1 次infer, KV_Caches 为 空
        #                   对于第 i > 1 次infer, KV_Caches 是 tensors shape as (1, i-1, d_dim), i-1 维包含 timestep 0 到 i-2

        #        position_ids:
        #           推理时阶段时, 对于第 1 次infer, position_ids 应该是 tensor([0]), 因为此时 tgt_dec_input 是 <bos>, KV_Caches 为 {}
        #           对于第 i > 1 次infer, position_ids = tensor([i-1]), 因为此时 tgt_dec_input position 是 i-1, 即 KV_Cacues 的 value 的第二维度
        
        # target input embedding
        # tgt_dec_input_embd: shape (batch_size, num_steps, num_hiddens)
        tgt_dec_input_embd = self.embedding(tgt_dec_input) * math.sqrt(self.num_hiddens) 
        # 使用 固定位置编码时, 避免位置编码的影响过大，所以放大input embeddings

        # target position embedding
        # 训练模式: input 的 timesteps 是从 0(bos) 到 num_steps-1
        if self.training:
            _, num_steps = tgt_dec_input.shape
            position_ids = torch.arange(0, num_steps, dtype=torch.int64, device=tgt_dec_input.device) # (num_steps,)
        # 推理模式: 对于第i次infer, input 的 timestep 就是 i-1, 而这个信息可以从 KV_Caches 的values 中的第二个维度(dim=1)得到
        else:
            position_ids = torch.tensor([ 0 if KV_Caches == {} else KV_Caches['0'].size(1) ],
                                        dtype=torch.int64, device=tgt_dec_input.device) # (1,)

        # input embeddings + position embedding
        tgt_query = self.dropout(tgt_dec_input_embd + self.pos_encoding(position_ids))

        # Decoder Block 的输入 tgt_query, src_enc_info, KV_Caches
        for blk in self.blks:
            # 循环过程中, 单次 blk 执行, 更新了 该 blk 对应的 KV_Caches 的 block-ID: tensor 的 kv对
            tgt_query, KV_Caches = blk(tgt_query, src_enc_info, KV_Caches)
        
        # 一次 infer forward 过程, KV_Caches 中的每个 key-value pair, 都被更新, 则 整个 KV_Caches 被更新

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
