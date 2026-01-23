from dataclasses import dataclass

@dataclass
class bertConfig:
    ## embedding layer configs
    num_hiddens:int
    vocab_size:int
    embd_p_drop:float
    ## position embedding configs
    use_abspos:bool
    seq_len:int
    ## encoder-block(bidirectional-attention+ffn) configs
    num_heads:int
    use_bias:bool
    ffn_num_hiddens:bool
    resid_p_drop:float
    ## number of encoder-block
    num_blks:int




__all__ = ["bertConfig"]