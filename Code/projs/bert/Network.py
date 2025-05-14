import torch.nn as nn
from torch import Tensor
import torch
from ...Modules._transformer import TransformerEncoderBlock
from ...Base.RootLayers.PositionalEncodings import LearnAbsPosEnc, TrigonoAbsPosEnc






# Encoder 组件: 所有预训练 encoder 组件应该干同一样事情: 把shape为 (batch_size, seq_length) 的序列数据, 转换为 (batch_size, seq_length, num_hiddens)
# 的 encoded tensor. 这个才叫 encoder
class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, seq_len, use_bias=True, **kwargs):
        super().__init__()
        # token embedding layer: (batch_size, seq_length)int64 of vocab_size --embedding--> (batch_size, seq_length, num_hiddens)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)

        # position embedding layer: (seq_length, )int64 of position ID --embedding--> (seq_length, num_hiddens)
        self.pos_encoding = LearnAbsPosEnc(max_possible_posNum=seq_len, num_hiddens=num_hiddens)
        # 输入序列已经被 pad/truncate 到同一长度, 并且额外 append 的 <cls> 和 <sep> token 也被计算在 seq_len 里了
        
        # dropout layer: dropout on token_embd + pos_embd
        self.dropout = nn.Dropout(dropout)

        # segment embedding layer: (batch_size, seq_length)int64 of 0/1 --embedding--> (batch_size, seq_length, num_hiddens)
        self.seg_embedding = nn.Embedding(2, num_hiddens)

        # encoder layer: token embd + pos embd + seg embd: (batch_size, seq_length, num_hiddens) --single encoder block-->
        # (batch_size, seq_length, num_hiddens) 
        self.blks = nn.Sequential()
        for i in range(num_blks):
            cur_blk = TransformerEncoderBlock(num_heads, num_hiddens, dropout, ffn_num_hiddens, use_bias)
            self.blks.add_module(f'blk{i+1}', cur_blk)
    

    def forward(self, tokens, segments, valid_lens):
        # tokens shape: (batch_size, seq_len)int64
        # segments shape: (batch_size, seq_len)01 int64
        # valid_lens: (batch_size,)
        X = self.token_embedding(tokens) + self.seg_embedding(segments)
        # positions shape: [0, 1, ..., seq_len-1] 1D tensor
        positions = torch.arange(0, X.size(1), dtype=torch.int64, device=X.device)
        X = self.dropout( X + self.pos_encoding(positions) )

        for blk in self.blks:
            X = blk(X, valid_lens)

        return X # 输出 shape: (batch_size, seq_len, num_hiddens)



# pretrain 任务组件
# 输入来自 encoder 组件的 tensor (batch_size, seq_len, num_hiddens), 以及 task 指定的 额外输入
# 输出 task 指定

# 1: Mask language model
# 额外输入 mask_positions, 即需要对 mask tokens 作预测 的 index 位置们
class MLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens):
        super().__init__()

        # 用一个 MLP 作预测, (*, d_dim) --mlp on last dim--> (*, vocab_size)
        self.mlp = nn.Sequential(nn.LazyLinear(num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.LazyLinear(vocab_size))
    
    # 使用 torch.gather(input, dim, index)
    # input shape: (d0, d1, ... di ..., dN), dim = i, index shape: (d0, d1, ... m, ..., dN)int64 of size(di)
    # 指定 在 第 i 维 收集, 除了第 i 维, 其他维度上 input 和 index shape 都相同
    # 输出output shape: (d0, d1, ... m, ..., dN)
    
    # 指定 在 第 i 维收集, 是指 输出output 在 某位置 position [ind_0, ind_i, ..., ind_i, ..., ind_N] 的值是这样确定的:
    # 首先从 index 得到 第 i 维的查找位置: index[position] = I, 然后从 input 的该位置 [ind_0, ind_i, ..., I, ..., ind_N] 找到 值
    def forward(self, token_embd, mask_positions):
        # token_embd shape: (batch_size, seq_len, d_dim) embed后的 sequence
        # mask_positions: (batch_size, num_masktks)int64 of 0-seq_len-1, 需要在 sequence 相应位置里 作pred 的 位置的 index值.
        # num_masktks 代表需要pred的token个数

        # 对于每个 batch 样本 i
        # 从 token_embd[i] (seq_len, d_dim) 中取出 索引为 mask_positions[i] (num_masktks,) 的那些行, 得到
        # (num_masktks, d_dim) 的结果，最终整合整个 batch, 得到
        # (batch_size, num_masktks, d_dim) 的输出

        # select_index_tensor: (batch_size, num_masktks) --> (batch_size, num_masktks, 1) --> (batch_size, num_masktks, d_dim)
        select_index_tensor = mask_positions.unsqueeze(2).expand(-1, -1, token_embd.size(2))

        # token_embd shape: (batch_size, seq_len, d_dim) --gather on dim 1--> (batch_size, num_masktks, d_dim)
        mask_tokens_embd = torch.gather(token_embd, 1, select_index_tensor)

        return self.mlp(mask_tokens_embd) # (batch_size, num_masktks, vocab_size)
    
        # 实际上 mask_positions 里, 0 代表 pad, 它会从 token_embd 中抽取 序列位置为 0 的token tensor 作预测, 而这个位置都是 <cls>
        # pad 过程中, position pad 0时, label 也会被 pad 一个tokenID. 但这部分的预测没有意义, 应该在loss中忽略










# pretrain 任务组件
# 输入来自 encoder 组件的 tensor (batch_size, seq_len, num_hiddens), 以及 task 指定的 额外输入
# 输出 task 指定

# 2: Next sentence pair model
# 不需要额外输入
class NSP(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        self.mlp = nn.Sequential(nn.LazyLinear(num_hiddens), nn.Tanh())
        self.head = nn.LazyLinear(2)
    
    def forward(self, token_embd):
        # token_embd shape: (batch_size, seq_len, d_dim) embed后的 sequence
        return self.head( self.mlp(token_embd[:, 0, :]) ) # output shape: (batch_size, 2)







# 组装 encoder + pretrain tasks --> combined loss
class BERT(nn.Module):
    def __init__(self, vocab_size, num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, seq_len, use_bias):
        super().__init__()
        self.encoder = BERTEncoder(vocab_size, num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, seq_len, use_bias)
        self.mlm = MLM(vocab_size, num_hiddens)
        self.nsp = NSP(num_hiddens)
    
    def forward(self, tokens, valid_lens, segments: Tensor|None, mask_positions: Tensor|None):
        '''
        对于 fine-tune 不一定需要的输入参数, 要设默认为 None 以不进入相关任务. 比如两个 pre-train 任务特需的输入
        '''
        # tokens: (batch_size, seq_len)int64 ot token ID. 已包含<cls>和<sep>
        # valid_lens: (batch_size,)

        # segments: (batch_size, seq_len)01 indicating seq1 & seq2
        # mask_positions: (batch_size, num_masktks) | None, None 代表当前 batch 不需要进入 MLM task, 只需要 NSP task
        embd_X = self.encoder(tokens, segments, valid_lens) # (batch_size, seq_len, num_hiddens)

        if mask_positions is not None:
            mlm_Y_hat = self.mlm(embd_X, mask_positions)# (batch_size, num_masktks, vocab_size)
        else:
            mlm_Y_hat = None

        if segments is not None:
            nsp_Y_hat = self.nsp(embd_X) # (batch_size, 2)
        else:
            nsp_Y_hat = None

        return embd_X, mlm_Y_hat, nsp_Y_hat