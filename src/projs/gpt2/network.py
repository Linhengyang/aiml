from ...core.nn_components.meta_frames import Encoder, AttentionDecoder, EncoderDecoder
from ...core.nn_components.root_layers.position_encoding import TrigonoAbsPosEnc
from ...core.nn_components.sub_modules._gpt2 import GPT2DecoderBlock
import torch.nn as nn
import math
import torch





class gpt2(AttentionDecoder):
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
            cur_blk = GPT2DecoderBlock(i, num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias)
            self.blks.add_module("block"+str(i), cur_blk)
        # TODO
        # embedding matrix or linear?
        self.dense = nn.Linear(num_hiddens, vocab_size)


    def forward(self, tgt_dec_input, KV_Caches=None):
        # train: tgt_dec_input shape: (batch_size, context_size)int64, timestep 从 0 到 context_size-1
        #        KV_Caches: None

        #        position_ids: 
        #           训练阶段时, position_ids 应该是 tensor([0, 1, ..., context_size-1])

        # 对于第i次infer: i = 1, 2, ..., context_size
        #        tgt_dec_input shape: (1, 1)int64, 其 timestep 是 i-1, 前向的output的 timestep 是 i
        #        input KV_Caches, which is dict with:
        #           keys: block_ind,
        #           values: 对于第 1 次infer, KV_Caches 为 空
        #                   对于第 i > 1 次infer, KV_Caches 是 tensors shape as (1, i-1, d_dim), i-1 维包含 timestep 0 到 i-2

        #        position_ids:
        #           推理时阶段时, 对于第 1 次infer, position_ids 应该是 tensor([0]), 因为此时 tgt_dec_input 是 <bos>, KV_Caches 为 {}
        #           对于第 i > 1 次infer, position_ids = tensor([i-1]), 因为此时 tgt_dec_input position 是 i-1, 即 KV_Cacues 的 value 的第二维度
        
        # target input embedding
        # tgt_dec_input_embd: shape (batch_size, context_size, num_hiddens)
        tgt_dec_input_embd = self.embedding(tgt_dec_input) * math.sqrt(self.num_hiddens) 
        # 使用 固定位置编码时, 避免位置编码的影响过大，所以放大input embeddings

        # target position embedding
        # 训练模式: input 的 timesteps 是从 0(bos) 到 context_size-1
        if self.training:
            _, context_size = tgt_dec_input.shape
            position_ids = torch.arange(0, context_size, dtype=torch.int64, device=tgt_dec_input.device) # (context_size,)
        # 推理模式: 对于第i次infer, input 的 timestep 就是 i-1, 而这个信息可以从 KV_Caches 的values 中的第二个维度(dim=1)得到
        else:
            position_ids = torch.tensor([ 0 if KV_Caches == {} else KV_Caches['0'].size(1) ],
                                        dtype=torch.int64, device=tgt_dec_input.device) # (1,)

        # input embeddings + position embedding
        tgt_query = self.dropout(tgt_dec_input_embd + self.pos_encoding(position_ids))

        # Decoder Block 的输入 tgt_query, KV_Caches
        for blk in self.blks:
            # 循环过程中, 单次 blk 执行, 更新了 该 blk 对应的 KV_Caches 的 block-ID: tensor 的 kv对
            tgt_query, KV_Caches = blk(tgt_query, KV_Caches)
        
        # 一次 infer forward 过程, KV_Caches 中的每个 key-value pair, 都被更新, 则 整个 KV_Caches 被更新

        #train: output[0] shape: (batch_size, context_size, vocab_size)tensor of logits,  timestep 从 1 到 context_size;
        #       output[1]: None
        #infer: output[0] shape: (1, 1, vocab_size)tensor of logits, 对于第i次infer, timestep 是 i;
        #       output[1]: dict of 
        #                   keys as block_indices
        #                   values as (1, i, d_dim) tensor, i 维 包含 timestep 0-i-1, 实际上就是 input KV_Caches 添加 timestep i-1 的 tgt_query
        return self.dense(tgt_query), KV_Caches