import torch.nn as nn
import torch
from src.core.blocks.transformer import TransformerEncoderBlock
from src.core.layers.position_encoding import LearnAbsPosEnc, TrigonoAbsPosEnc
from .config_bert import bertConfig

# Encoder 组件: 所有预训练 encoder 组件应该干同一样事情: 把shape为 (batch_size, seq_length) 的序列数据, 转换为 (batch_size, seq_length, num_hiddens)
# 的 encoded tensor. 这个才叫 encoder
class BERTEncoder(nn.Module):
    def __init__(self, config:bertConfig):
        super().__init__()
        # token embedding layer: (batch_size, seq_length)int64 of vocab_size --embedding--> (batch_size, seq_length, num_hiddens)
        self.token_embedding = nn.Embedding(config.vocab_size, config.num_hiddens)

        # position embedding layer: (seq_length, )int64 of position ID --embedding--> (seq_length, num_hiddens)
        if config.use_abspos:
            self.pos_encoding = TrigonoAbsPosEnc(config.num_hiddens)
        else:
            self.pos_encoding = LearnAbsPosEnc(config.seq_len, config.num_hiddens)
        # 输入序列已经被 pad/truncate 到同一长度, 并且额外 append 的 <cls> 和 <sep> token 也被计算在 seq_len 里了
        
        # dropout layer: dropout on token_embd + pos_embd
        self.dropout = nn.Dropout(config.embd_p_drop)

        # segment embedding layer: (batch_size, seq_length)int64 of 0/1 --embedding--> (batch_size, seq_length, num_hiddens)
        self.seg_embedding = nn.Embedding(2, config.num_hiddens)

        # encoder layer: token embd + pos embd + seg embd: (batch_size, seq_length, num_hiddens) --single encoder block-->
        # (batch_size, seq_length, num_hiddens) 
        self.blks = nn.Sequential()
        for i in range(config.num_blks):
            cur_blk = TransformerEncoderBlock(
                config.num_heads,
                config.num_hiddens,
                config.resid_p_drop,
                config.ffn_num_hiddens, 
                config.use_bias
                )
            self.blks.add_module(f'blk{i+1}', cur_blk)
    

    def forward(self, tokens, valid_lens, segments):
        # tokens shape: (batch_size, seq_len)int64
        # valid_lens: (batch_size,)
        # segments shape: (batch_size, seq_len)01 int64
        
        X = self.token_embedding(tokens) + self.seg_embedding(segments) # X shape: (batch_size, seq_len, num_hiddens)
        # positions shape: [0, 1, ..., seq_len-1] 1D tensor
        positions = torch.arange(0, X.size(1), dtype=torch.int64, device=X.device)
        # X(batch_size, seq_len, num_hiddens) broadcast + posenc(seq_len, num_hiddens)
        X = self.dropout( X + self.pos_encoding(positions) )

        for blk in self.blks:
            X = blk(X, valid_lens)

        return X # 输出 shape: (batch_size, seq_len, num_hiddens)
    


__all__ = ["BERTEncoder"]