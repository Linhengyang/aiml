import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math

def masked_softmax(S, valid_lens):
    '''
    inputs: S, valid_lens
        S: 3-Dtensor, shape: (batch_size, n_queries, n_kvpairs);
        valid_lens: 1-D or 2—D tensor, shape: (batch_size,) or (batch_size, n_queries)
        (if len=0 in valid_lens, it means average all Vs in QKV pool)
    
    returns: convex weight tensor with shape (batch_size, n_queries, n_kvpairs), denoted as W
        W[sample_idx, query_idx, :] is a 1-D tensor of convex weight distribution(sum to 1 and non-negative).

    explains:
        for sample i,
            if valid_lens is 1-D tensor, W[i][:, k] are zeros when k > valid_lens[i]
            if valid_lens is 2-D tensor, W[i][j, k] are zeros when k > valid_lens[i, j], here j is query_idx
    
    将S打分矩阵的部分元素换成无穷小(indexput操作), 然后再作softmax操作. 在torch框架下,这两个操作都是梯度可传的.
    1. 保证了invalid位置的元素, 不会参与计算下一层valid位置的元素.(因为softmax操作. softmax(无穷小)=0, 使得invalid位置元素的权重为0)
    2. 保证了invalid位置的元素, 不论结果如何, 都不会在BP中贡献更新组成运算它们的参数.(因为indexput操作)
    众所周知, masked_softmax是为了完成两个目标:
        1. 对source sequence作token是否valid甄别, 使得valid token仅由valid tokens表征(纯洁性)
        2. 对target sequence作token是否自回甄别, 使得current token仅由past tokens表征(纯洁性)
    这种对纯洁性的保证, 是从两个方面保证的， 即 1. invalid位置的元素不能参与计算下一层valid位置的元素; 2. invalid位置的元素无论结果如何, 不能在bp中贡献更新运算它们的参数
    '''

    # 确定2-D tensor的mask操作。这里X是2-D tensor, valid_len是1-D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        
        mask = torch.arange(maxlen, dtype=torch.float32, device=X.device).unsqueeze(0) < valid_len.unsqueeze(1)
        X[~mask] = value # indexput 操作是梯度可传的. 被put的位置在之后的BP反传中,将不会贡献更新组成运算它们的参数
        return X
    
    # 将S和valid_lens分别转化为2-D tensor和1-D tensor
    if valid_lens is None: #如果不输入valid_lens，那么所有元素参与权重化
        return nn.functional.softmax(S, dim=-1) # nn.f.softmax操作是梯度可传的
    else:
        shape = S.shape # 保存S的shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1]) # 拉长，返回还是是1-D tensor
        else:
            valid_lens = valid_lens.reshape(-1) # 摊平，返回1-D tensor
        # 将S转化为2-D tensor, last axis不变
        S = _sequence_mask(S.reshape(-1, shape[-1]), valid_lens, value=-1e20)
        return nn.functional.softmax(S.reshape(shape), dim=-1) # nn.f.softmax操作是梯度可传的

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
    def forward(self, Q_batch, K_batch, V_batch, valid_lens=None):
        # m:n_query, q:query_size; n: n_kv, k:key_size; v:value_size
        # Q_batch: batch of shape(m, q); K_batch: batch of shape(n, k); V_batch: batch of shape(n, v)
        batch_size, m, q = Q_batch.shape
        _, n, k = K_batch.shape
        # Q_batch_tilda shape(batch_size, m, h), K_batch_tilda shape(batch_size, n, h)
        Q_batch_tilda, K_batch_tilda = self.W_q(Q_batch), self.W_k(K_batch)
        # S_batch shape是(batch_size, m, n, h)
        S_batch = Q_batch_tilda.unsqueeze(2).expand(-1, -1, n, -1) + K_batch_tilda.unsqueeze(1).expand(-1, m, -1, -1)
        Scores = self.W_v(torch.tanh(S_batch)).squeeze(-1) # shape是(batch_size, m, n, 1) --> # shape是(batch_size, m, n)
        self.attention_weights = masked_softmax(Scores, valid_lens) # shape是(batch_size, m, n)
        return torch.bmm(self.dropout(self.attention_weights), V_batch) # 返回的shape是(batch_size, m, v)

class ScaledDotProductAttention(nn.Module):
    '''
    args: dropout
        dropout: dropout rate. Regularization on the attention weight matrices
    
    inputs: Q_batch, K_batch, V_batch, valid_lens(optional)
        Q_batch's shape: (batch_size, n_queries, qk_size)
        K_batch's shape: (batch_size, n_kvs, qk_size)
        V_batch's shape: (batch_size, n_kvs, value_size)
        valid_lens(optional)'s shape: (batch_size,) or (batch_size, n_queries)
    
    returns: denoted as O
        O's shape: (batch_size, n_queries, value_size)
    
    explains:
        returns W @ V where W is attention pool of (Q, K)
        Scaled Dot Production on queries and keys to attention pool for weight matrices W (batch_size, n_queries, n_kvs)
        Convex combinations of Values based on weight matrices are returned
    '''
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q_batch, K_batch, V_batch, valid_lens=None):

        assert Q_batch.size(-1) == K_batch.size(-1), \
            f'query_size {Q_batch.size(-1)} not equal to key_size {K_batch.size(-1)}'
        
        d = Q_batch.size(-1)
        S_batch = torch.bmm(Q_batch, K_batch.permute(0, 2, 1)) / math.sqrt(d)

        self.attention_weights = masked_softmax(S_batch, valid_lens)

        return torch.bmm( self.dropout(self.attention_weights), V_batch )

def transpose_qkv(X, num_heads):
    h = num_heads
    batch_size, n, _ = X.shape
    return X.permute(0,2,1).reshape(batch_size, h, -1, n).permute(0,1,3,2).reshape(batch_size*h, n, -1)

def transpose_o(X, num_heads):
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
            1. H linear projections on Q K V whom projected to num_hiddens // h dimensions, which stands for H heads
            2. For every head's result, perform scaled-dot-prod attention pool
            3. Assemble H attenion-poolings' output, to have a result with num_hiddens dimensions. A final num_hiddens to 
               num_hiddens linear project is followed
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
    
    def forward(self, Q_batch, K_batch, V_batch, valid_lens=None):
        Q = transpose_qkv(self.W_q(Q_batch), self.h)
        K = transpose_qkv(self.W_k(K_batch), self.h)
        V = transpose_qkv(self.W_v(V_batch), self.h)
        if valid_lens is not None:
            valid_lens = valid_lens.repeat_interleave(self.h, dim=0)
        O = self.attention(Q, K, V, valid_lens)
        O = transpose_o(O, self.h)
        return self.W_o(O)
