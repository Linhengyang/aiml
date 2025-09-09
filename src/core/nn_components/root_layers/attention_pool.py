import torch
from torch import nn
import math
from ...base.functions.mask import mask_on_last_dim





    
# 将S打分矩阵的部分元素换成无穷小(index-set, or slice-copy), 然后再作softmax操作.
# 在torch框架下,这两个操作都是梯度可传的, grad_fn 分别是 CopySlice 和 softmax

# 1. 保证了invalid位置的元素, 不会参与计算下一层valid位置的元素.(因为softmax操作. softmax(无穷小)=0, 使得invalid位置元素的权重为0)
# 2. 保证了invalid位置的元素, 不论结果如何, 都不会在BP中贡献更新组成运算它们的参数.(因为 slice-copy 操作)

# masked_softmax是为了完成两个目标:
#     1. 对source sequence作token是否valid甄别, 使得valid token仅由valid tokens表征.
#     2. 对target sequence作token是否自回甄别, 使得current token仅由past tokens生成

# masked_softmax 生成了 shape 为(batch_size, n_queries, n_kvpairs) 的 W, 和 V (batch_size, n_kvpairs, v_size) 作 bmm 乘法, 即 batch_size 个
# (n_queries, n_kvpairs) @ (n_kvpairs, v_size) 矩阵乘法, 结果是 n_queries 个 V 行向量的凸线性组合, shape 为 (n_queries, v_size)

# 从单条 query 的结果来看, Q (batch_size, n_queries, qk_size) 中的单条query (1, 1, qk_size) 与 K (batch_size, n_kvpairs, qk_size) 转置的 单样本
# (1, qk_size, n_kvpairs) 生成 单条logits (1, 1, n_kvpairs)。经过mask_softmax操作后, 前 valid 部分形成 单条凸组合分布权重，后 invalid 部分权重为 0

# 即: n_queries 次(1, 1, qk_size) @ (1, qk_size, n_kvpairs) --> n_queries 条凸组合分布权重 (1, 1, n_kvpairs)
# mask_softmax只保留了 n_kvpairs 中前valid部分, 所以 K的单样本 (1, n_kvpairs, qk_size) 中只有前 valid 部分参与运算, 后 invalid 部分被舍弃
# 单条凸组合分布权重 (1, 1, n_kvpairs) @ V的单样本 (1, n_kvpairs, v_size)，得到的 (1, 1, v_size) 中只包含了 V 的单样本中 前 valid 部分.

# 综上, scaled-dot-production with masked-softmax, 即 masked_softmax(Q @ K', valid_lens) @ V 操作中, 
# Q: (batch_size, n_queries, qk_size) @ K: (batch_size, n_kvpairs, qk_size)  --transpose_matprod--> S: (batch_size, n_queries, n_kvpairs)
# --masked_softmax_with_valid_lens(batch_size, n_queries)--> W:  (batch_size, n_queries, n_kvpairs) @ V: (batch_size, n_kvpairs, v_size)
# = output: (batch_size, n_queries, v_size)
# Q中 所有 query 分别和 K中 valid 部分 得到了 W中 valid 部分, 然后和 V中 valid 部分 得到了结果。简而言之, valid_lens 作用在 K 和 V 的 n_kvpairs 维度
# mased_softmax with valid_lens 使得 在 QkV attention 计算中, K 和 V 只有 valid 部分参与了运算.
# Q: (batch_size, n_queries, qk_size) 代表了 (batch_size, n_queries)次 (1,1, qk_size) 查询. 查询结果output: (batch_size, n_queries, v_size)
# valid_lens 不能作用在 Q 和 output 中 的 n_queries 维度.


def masked_softmax(S, valid_lens, where_valid='left'):
    '''
    inputs:
        S: 3-Dtensor, shape: (batch_size, n_query, n_logits);
        valid_lens: 1-D or 2—D tensor, shape: (batch_size,) or (batch_size, n_query) or None
        where_valid: left or right.
            left: means valid_lens is counting from index 0
            right: means valid_lens is counting from index -1

    returns: convex weight tensor with shape (batch_size, n_query, n_logits), denoted as W
        W[sample_idx, query_idx, :] is a 1-D tensor of convex weight distribution(sum to 1 and non-negative).
        对每个样本而言, 返回 n_query 个 凸组合权重
    '''

    if valid_lens is None: #如果不输入valid_lens，那么所有元素参与权重化
        return nn.functional.softmax(S, dim=-1)
    
    elif valid_lens.dim() == 1: # valid_lens: (batch_size, ) --> (batch_size, 1) --> (batch_size, n_query)
        valid_lens = torch.repeat_interleave(valid_lens.unsqueeze(1), repeats=S.shape[1], dim=1)

    elif valid_lens.dim() == 2: # valid_lens: (batch_size, n_query)
        assert valid_lens.shape == S.shape[:-1], f"valid_lens.shape {valid_lens.shape} not match with S shape {S.shape}"

    else:
        raise ValueError(f'wrong valid_lens')
    
    # 让 non-valid 部分为 True, 以此形成的 mask tensor 可以给 non-valid 部分填入 negative inf

    if where_valid == 'left': # 当左边部分是valid时, 右边部分是non-valid，要在mask tensor中设为True
        mask_flag = False
    elif where_valid == 'right': # 当右边部分是valid时, 左边部分是non-valid，要在mask tensor中设为True
        mask_flag = True
    else:
        raise ValueError(f'wrong where_valid')

    mask = mask_on_last_dim(S.shape[-1], valid_lens, mask_flag)

    # mask tensor shape: (batch_size, n_query, n_kv)

    S[mask] = -1e20 # non-valid 部分填入 负无穷, 这样在 softmax 操作中被消去. index-put操作梯度可传

    return nn.functional.softmax(S, dim=-1)




class AdditiveAttention(nn.Module):
    '''
    args: num_hiddens, dropout
        num_hiddens: hidden size for Query & Key's linear-projecting aiming to add
        dropout: dropout rate. Regularization on the attention weight matrices
    
    inputs: Q_batch, K_batch, V_batch, valid_lens(optional)
        Q_batch's shape: (batch_size, n_queries, query_size)
        K_batch's shape: (batch_size, n_kvs, key_size)
        V_batch's shape: (batch_size, n_kvs, value_size)
        valid_lens(optional)'s shape: (batch_size,) or (batch_size, n_queries)
    
    returns: denoted as O
        O's shape: (batch_size, n_queries, value_size)
    
    explains:
        returns W @ V where W = attention pool of (Q, K)
        MLP of additive operation on queries and keys to attention pool for weight matrices W (batch_size, n_queries, n_kvs)
        Convex combinations of Values based on weight matrices are returned
    '''
    def __init__(self, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q_batch, K_batch, V_batch, valid_lens=None, where_valid='left'):
        # Q_batch: (batch_size, n_query, q_size); K_batch: (batch_size, n_kv, k_size); V_batch: (batch_size, n_kv, v_size)
        _, n_query, _ = Q_batch.shape
        _, n_kv, _ = K_batch.shape

        # Q_batch_tilda shape(batch_size, n_query, h), K_batch_tilda shape(batch_size, n_kv, h)
        Q_batch, K_batch = self.W_q(Q_batch), self.W_k(K_batch)

        # Q_batch: (batch_size, n_query, h) --> (batch_size, n_query, 1, h) --> (batch_size, n_query, n_kv, h)
        # K_batch: (batch_size, n_kv, h) --> (batch_size, 1, n_kv, h) --> (batch_size, n_query, n_kv, h)
        S_batch = Q_batch.unsqueeze(2).expand(-1, -1, n_kv, -1) + K_batch.unsqueeze(1).expand(-1, n_query, -1, -1)
        # S_batch: (batch_size, n_query, n_kv, h)

        # S_batch: (batch_size, n_query, n_kv, h) --> (batch_size, n_query, n_kv, 1) --> (batch_size, n_query, n_kv)
        Scores = self.W_v(torch.tanh(S_batch)).squeeze(-1)

        # Scores: (batch_size, n_query, n_kv), valid_lens 指定了每条 query 里的 valid area:(batch_size,) or (batch_size, n_query)
        self.attention_weights = masked_softmax(Scores, valid_lens, where_valid)

        # W: (batch_size, n_query, n_kvs) @ V:  (batch_size, n_kvs, value_size) ->  (batch_size, n_query, value_size) 
        return torch.bmm(self.dropout(self.attention_weights), V_batch)





class ScaledDotProductAttention(nn.Module):
    '''
    args: dropout
        dropout: dropout rate. Regularization on the attention weight matrices
    
    inputs: Q_batch, K_batch, V_batch, valid_lens(optional)
        Q_batch's shape: (batch_size, n_query, qk_size)
        K_batch's shape: (batch_size, n_kvs, qk_size)
        V_batch's shape: (batch_size, n_kvs, v_size)
        valid_lens(optional)'s shape: (batch_size,) or (batch_size, n_query)
    
    returns: denoted as O
        O's shape: (batch_size, n_query, v_size)
    
    explains:
        Q(batch_size, n_query, qk_size), K(batch_size, n_kvs, qk_size) --相似度计算--> W(batch_size, n_query, n_kvs)
        
        Scaled Dot Production on queries and keys to attention pool for weight matrices W (batch_size, n_queries, n_kvs)
        Convex combinations of Values based on weight matrices are returned
    
    query/key/value 都有各自的数量和维度. 其中 query 的数量自由决定, 但是其维度要和key相同(毕竟要计算query和key之间的相似度)
    value的维度自由决定, 但是其数量要和key相同(key决定了其对应value在最终输出结果中的重要程度)
    
    积式注意力 ScaledDotProductAttention 简单地根据 每条 query和不同keys之间地相似度, 决定了每个key对应的value的权重, 组合出最后的结果
    最终由 n_queries 条结果
    '''
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q_batch, K_batch, V_batch, valid_lens=None, where_valid='left'):

        assert Q_batch.size(-1) == K_batch.size(-1), \
            f'query_size {Q_batch.size(-1)} not equal to key_size {K_batch.size(-1)}'
        
        d = Q_batch.size(-1) # scale 相似度计算中因为维数过大引起的数值不稳定

        # Q: (batch_size, n_query, qk_size) @ K: (batch_size, n_kvs, qk_size) 转置 -> (batch_size, n_query, n_kvs)
        S_batch = torch.bmm(Q_batch, K_batch.permute(0, 2, 1)) / math.sqrt(d) # 本质是 Q 和 K 的相似度计算

        self.attention_weights = masked_softmax(S_batch, valid_lens, where_valid) # 注意力权重shape (batch_size, n_query, n_kvs)

        # W: (batch_size, n_query, n_kvs) @ V:  (batch_size, n_kvs, v_size) ->  (batch_size, n_query, v_size) 
        return torch.bmm( self.dropout(self.attention_weights), V_batch )




def transpose_qkv(X, num_heads):
    '''
    X shape: (batch_size, seq_length, num_hiddens)
    transpose route: --> (batch_size, num_hiddens, seq_length) --> (batch_size, num_heads, dim_per_head, seq_length)
                     --> (batch_size, num_heads, seq_length, dim_per_head) --> (batch_size*num_heads, seq_length, dim_per_head)
    '''
    h = num_heads
    batch_size, n, _ = X.shape
    return X.permute(0,2,1).reshape(batch_size, h, -1, n).permute(0,1,3,2).reshape(batch_size*h, n, -1)




def transpose_o(X, num_heads):
    '''
    X shape: (batch_size*num_heads, seq_length, dim_per_head)
    transpose route: --> (batch_size*num_heads, dim_per_head, seq_length) --> (batch_size, num_heads*dim_per_head, seq_length)
                     --> (batch_size, seq_length, num_heads*dim_per_head)
    '''
    h = num_heads
    prod_batchsize_h, m, _ = X.shape
    batch_size = prod_batchsize_h // h
    return X.permute(0,2,1).reshape(batch_size, -1, m).permute(0,2,1)






class MultiHeadAttention(nn.Module):
    '''
    args: num_heads, num_hiddens, dropout
        num_heads: number of heads (num_heads | num_hiddens), see explains
        num_hiddens: number of hiddens (num_heads | num_hiddens), see explains
        dropout: dropout rate. Regularization on the attention weight matrices
    
    inputs: Q_batch, K_batch, V_batch, valid_lens(optional)
        Q_batch's shape: (batch_size, n_queries, query_size)
        K_batch's shape: (batch_size, n_kvs, key_size)
        V_batch's shape: (batch_size, n_kvs, value_size)
        valid_lens(optional)'s shape: (batch_size,) or (batch_size, n_queries)
    
    returns: denoted as O
        O's shape: (batch_size, n_queries, num_hiddens)
    
    explains:
        After multiple linear projections of Q K V, assemble the scaled-dot-prod attention pools of these projections,
        and a final linear project is followed.
        In detail:
            1. H linear projections on Q K V whom projected to num_hiddens // H dimensions, which stands for H heads
            2. For every head's result, perform scaled-dot-prod attention pool
            3. Assemble H attenion-poolings' output, to have a result with num_hiddens dimensions. A final num_hiddens to 
               num_hiddens linear project is followed

    多头注意力:
        单头注意力是指 对 QKV 作各自线性映射(至相同维度 num_hiddens/H )后, 作 ScaledDotProductAttention 后得到 (batch_size, n_queries, num_hiddens/H)
    H 个这样的单头注意力的结果, 拼起来是一个 (batch_size, n_queries, num_hiddens) 的结果. 再follow一个 num_hiddens -> num_hiddens 的线性映射
    '''
    def __init__(self, num_heads, num_hiddens, dropout, use_bias=False, **kwargs):
        assert num_hiddens % num_heads == 0, 'output dim of multihead att-pool is not divisible by number of heads'
        super().__init__(**kwargs)
        self.h = num_heads
        self.W_q = nn.LazyLinear(num_hiddens, bias=use_bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=use_bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=use_bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=use_bias)
        self.attention = ScaledDotProductAttention(dropout)
    
    def forward(self, Q_batch, K_batch, V_batch, valid_lens=None, where_valid='left'):
        # Q/K/V 都被split成多个head: (batch_size*num_heads, seq_length, dim_per_head)
        # attention计算时, q/k shape (B, n_query, q_size)/(B, n_kv, k_size) --> attn_w (B, n_query, n_kv)
        # 相关性计算只存在于dim 1和2, dim 0（B）之间没有相关性计算, dim 0维度之间仍然是独立的, 即 attn_w[i] = attn_func( q[i], k[i] )
        # 由此, head维度被合并到batch_size维度后, 在attention计算时head之间是独立的.
        Q = transpose_qkv(self.W_q(Q_batch), self.h)
        K = transpose_qkv(self.W_k(K_batch), self.h)
        V = transpose_qkv(self.W_v(V_batch), self.h)
        if valid_lens is not None:
            valid_lens = valid_lens.repeat_interleave(self.h, dim=0)
        O = self.attention(Q, K, V, valid_lens, where_valid)
        O = transpose_o(O, self.h)
        return self.W_o(O)




# F.scaled_dot_product_attention 的 api 逻辑, 实现带因果遮罩 casual_mask 的 multi-head attention layer
# (function) def scaled_dot_product_attention(
#     query: Tensor,
#     key: Tensor,
#     value: Tensor,
#     attn_mask: Tensor | None = None,
#     dropout_p: float = 0,
#     is_causal: bool = False,
#     scale: float | None = None,
#     enable_gqa: bool = False
# ) -> Tensor
# casual mask提供是否缓存选择:
#   缓存则 register_buffer 一个 max_context_size 的上三角 casual mask, 以便在 forward 中slice取出每条query相应的未来位置
#   不缓存则 在 forward 中按需构造一个 casual mask, 来制作每条query相应的未来位置, 这样就避免了forward中存在max_context_size
# attention中提供是否rope选择:
#   yes则在计算self-attention时, apply_rope on q/k， no则在计算self-attention时，直接计算 qk


# 由此，CasualMHA 在forward过程可以不依赖 max_context_size: casual_mask可以根据query_seq_length和kv_seq_length按需构造的, RoPE也不依赖
# 实际上，CasualMHA 还实现了 input sequence 变长：因为 RoPE 和 attention weights 都支持变长

# 不过, 在实际Model层面, 仍然会设定max_context_size参数，并保证所有input sequence 的长度都不大于这个值。原因如下:
#   1. 在 CasualMHA层, casual_mask 和 attention weights在训练时, 涉及 O(query_seq_length^2) 的显存空间占用. 如果不限制 query_seq_length，
#      那么在训练时若输入了过长的 input sequence，可能会导致OOM。这样在 CasualMHA 层避免了过大的 attention weights 矩阵。
#   2. 很多模型在制作train dataset时，会切分长文档/拼接短文档，成固定长度的chunk。这里的固定长度就约等于模型的max_context_size，因为限制了外推能力.
#      现代很多LLM的训练策略是curriculum-style: 训练epoch早期使用短序列，后期用长序列，训练时sequence length是动态变化的。
#   3. RoPE的无限拓展只是理论上的，实际上theta=position_index*频率信号，当position很大时会失真。现代模型多会启用RoPE scaling，而这里
#      RoPE scaling 依赖 max_context_size

class CasualMHA(nn.Module):
    pass