import warnings
warnings.filterwarnings("ignore")
import re
from Code.projs.transformer.Dataset import *
import os
from Code.Utils.Text.Vocabulize import Vocab
import math

if __name__ == "__main__":
    # base_path = "../../data"
    # proj_folder = "text_translator/fra-eng"
    # data_file = "fra.txt"

    # fname = os.path.join(base_path, proj_folder, data_file)
    # print(fname)

    # # test vocab

    # ## token 2D list
    # text = read_text2str((fname)) # raw text
    # text = preprocess_space(text, True, normalize_whitespace=False) # 保留 /t, 因为source和target是用 /t 分开的

    # source, target = tokenize_seq2seq(text, line_tokenize_simple, None, num_examples=10) # 使用 line_tokenize_simple 作 token化. 此时不需要 symbols

    # print("token 2d list")
    # print("source\n", source)
    # print("target\n", target)

    # ## make vocab: 
    # src_vocab = Vocab(source, min_freq = 0, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # print("tokens' id as")
    # print('<pad>', src_vocab['<pad>'])
    # print('<bos>', src_vocab['<bos>'])
    # print('<eos>', src_vocab['<eos>'])
    # print('go', src_vocab['go'])

    # print("ids' token as")
    # print("1", src_vocab.to_tokens(1))
    # print("0", src_vocab.to_tokens(0))

    # ## vocab supports hierarchy of tokens
    # print("token sequence as")
    # print("go run who ! . <unk> hi go . <pad> as ", src_vocab[["go", "run", "who", "!", ".", "<unk>", "hi", "go", ".", "<pad>"]])


    # # test Tensor

    # ## id 映射
    # lines = [src_vocab[line] + [src_vocab['<eos>']] for line in source]
    # print("mapped tokens as", source)
    # print("mapped token ids as", lines)

    # ## truncate or pad: 截断/补齐 每一行 line 到 给定长度
    # num_steps = 5
    # aligned_lines = [truncate_pad(l, num_steps, src_vocab['<pad>']) for l in lines] # 2D list: shape as (num_lines, num_steps)
    # print("aligned_lines as\n", aligned_lines)

    # array = torch.tensor(aligned_lines)
    # print("array as\n", array)

    # ## valid lens: 需要保存 每一行 line 中 非 pad 的token 的数量信息: (num_lines, )
    # valid_lens = (array != src_vocab['<pad>']).type(torch.int32).sum(1)
    # print("valid lens(non-padding) as \n", valid_lens)


    # # test torch dataset
    # class testTorchDataset(torch.utils.data.Dataset):
    #     '''
    #     torch Dataset 要继承自 torch.utils.data.Dataset 类, 并提供三个 内置函数
    #     1、__init__方法: 读取并定义 ALL Data into tensor data 形式
    #     2、__getitem__方法: 有一个固定 param index, 以 index 定义 单个样本 single example of index 如何从该 Dataset 对象中释出
    #     3、__len__方法: 返回 ALL Data 的 data size
        
    #     __init__方法确认了 ALL Data. __getitem__方法确认了 single datapoint. 它和__len__方法一起帮助 torch 的 dataiter工具从 ALL Data中生成 Data Batch
    #     '''

    #     def __init__(self, path, num_steps, num_examples=None):
    #         super().__init__() # torch dataset 继承 torch.utils.data.Dataset
    #         (X, X_valid_lens, Y, Y_valid_lens), (src_vocab, tgt_vocab) = build_dataset_vocab(path, num_steps, num_examples)
    #         # X, Y: (num_examples, num_steps) padding sequence data
    #         # X_valid_lens, Y_valid_lens: (num_examples,) valid length info data
    #         # src_vocab, tgt_vocab: mapping vocab for source and target text

    #         self._data_size = Y.shape[0]

    #         # bos: tensor with shape (num_examples, 1) to append before Y (num_examples, num_steps)
    #         bos = torch.tensor( [tgt_vocab['<bos>']] * self._data_size, device=Y.device ).reshape(-1, 1)
    #         # concatenate bos and Y's num_steps-1 on dim 1 to create dec_X (num_examples, num_steps)
    #         dec_X = torch.cat([bos, Y[:, :-1]], dim=1)

    #         self._net_inputs = (X, dec_X, X_valid_lens)
    #         self._loss_inputs = (Y, Y_valid_lens)

    #         self._src_vocab = src_vocab
    #         self._tgt_vocab = tgt_vocab
        

    #     def __getitem__(self, index):
    #         return (tuple(tensor[index] for tensor in self._net_inputs),
    #                 tuple(tensor[index] for tensor in self._loss_inputs))


    #     def __len__(self):
    #         return self._data_size
        
    #     @property
    #     def src_vocab(self):
    #         return self._src_vocab
        
    #     @property
    #     def tgt_vocab(self):
    #         return self._tgt_vocab
        
        
    ## torch dataset and data loader
    # trainset = testTorchDataset(fname, num_steps=5, num_examples=10)

    ## print batch data
    # train_iter = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=True)
    # for x, y in train_iter:
    #     print("epoch 1")
    #     print('x as\n', x)
    #     print('y as\n', y)

    # for x, y in train_iter:
    #     print("epoch 2")
    #     print('x as\n', x)
    #     print('y as\n', y)

    ## allocate batch data to another device

    ## collate_fn param for DataLoader
    ## Dataset对象中, __getitem__()返回的是 index 对应的单个 data point: data_ind, label_ind
    ## Dataset对象传入 DataLoader对象中时, collate_fn 参数指定了一个 如何处理 batch_list 的函数, 以处理 batch_size 个 data point
    ## collate_fn 默认是 default_collate 函数, 它将 batch_list(batch_size个data/label point组成的 list), 处理成shape为(batch_size,...)的DATA, 
    ## 和shape为(batch_size,...)的LABEL 两个 batch tensor dataset

    ## collate_fn可以是自定义 batch 处理函数. 它应该满足: 以 batch_list(batch_size个__getitem__定义的datapoint组成的list)为输入, 输出batch处理后的结果
    # from torch.utils.data.dataloader import default_collate
    # device_cuda = torch.device('cuda')


    # def move_to_cuda(batch_list): 
    #     '''
    #     __getitem__ 返回 datapoint of index:
    #         (tuple(tensor[index] for tensor in [X, dec_X, X_valid_lens]), tuple(tensor[index] for tensor in [Y, Y_valid_lens]))
    #         即:
    #         (X[index], dec_X[index], X_valid_lens[index]), (Y[index], Y_valid_lens[index])
    #     经过 default_collate(batch_list), 返回 batch data:
    #         (tuple(tensor[batch::] for tensor in [X, dec_X, X_valid_lens]), tuple(tensor[batch::] for tensor in [Y, Y_valid_lens]))
    #         即:
    #         (X[batch::], dec_X[batch::], X_valid_lens[batch::]), (Y[batch::], Y_valid_lens[batch::])
    #     逐一move到cuda上
    #     '''
        
    #     # result = []
    #     # for x_ in default_collate(batch_list):
    #     #     res_ = tuple()
    #     #     for t in x_:
    #     #         res_ += (t.to(device_cuda),)
    #     #     result.append(res_)
    #     # return tuple(result)
    #     (X_batch, dec_X_batch, X_valid_lens_batch), (Y_batch, Y_valid_lens_batch) = default_collate(batch_list)

    #     X_batch = X_batch.to(device_cuda)
    #     dec_X_batch = dec_X_batch.to(device_cuda)
    #     X_valid_lens_batch = X_valid_lens_batch.to(device_cuda)
    #     Y_batch = Y_batch.to(device_cuda)
    #     Y_valid_lens_batch = Y_valid_lens_batch.to(device_cuda)

    #     return (X_batch, dec_X_batch, X_valid_lens_batch), (Y_batch, Y_valid_lens_batch)
    
    
    # train_iter = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True, collate_fn=move_to_cuda)
    # for x, y in train_iter:
    #     print("epoch 3")
    #     print('x in cuda\n', x[0].device)
    #     print('y in cuda\n', y[0].device)


    # network: transformer = encoder + decoder
    
    ## encoder

    ## mask_softmax
    ## 制作一个 mask 以完成 mask_softmax 操作: 对于一个 (batch_size, n_queries, n_kvs) 的 score matrix batch, 总共有 batch_size*n_queries 条query
    ## 对每条query 都只取 靠前的部分 score logits 参与softmax 计算。这个根据 不同 query 决定的 valid lens 储存在一个 (batch_size, n_queries) 的矩阵里 
    ## 需要一个 mask tensor(batch_size, n_queries, n_kvs) , 其元素是True/False. 在 batch_i, n_queries_j 的位置, 前 valid_len 个 为True, 剩下的为False
    ## batch_i, n_queries_j 的 valid_len，是由 valid_lens (batch_size, n_queries) 的(i,j)元 确定的
    ## 原理就是 不断地让 [0, 1, ..., n_kvs-1] 和 valid_lens[i,j] 元素作 对比是否小于 的bool操作
    maxlen = 5

    ## (-1, n_kvs) broadcast 机制
    query_indices = torch.arange(maxlen, dtype=torch.float32)
    print(query_indices)

    ## (batch_size, n_queries)
    valid_lens = [[3, 4, 1, 0],
                  [2, 1, 0, 1],
                  [1, 4, 5, 5]]
    
    ## (batch_size, n_queries, 1)
    valid_lens = torch.tensor(valid_lens).unsqueeze(2)
    print(valid_lens)

    ## (batch_size, n_queries, n_kvs) broadcast机制
    mask = query_indices < valid_lens
    print( mask )

    ## valid_lens 也可以是 (batch_size,) 的向量. 此时 对于 batch_i, 所有 query 的valid长度 都由 valid_lens[i] 决定
    ## (batch_size, 1)
    valid_lens = torch.tensor([1, 3, 5]).unsqueeze(1)
    print(valid_lens)
    ## (batch_size, n_queries)
    
    valid_lens = torch.repeat_interleave( valid_lens, repeats=4, dim=1)
    print(valid_lens)

    ## index-put 操作 对 梯度反向传播的影响
    test_tensor = torch.tensor([1., 2., 3., 4.], requires_grad=True)
    
    # power 2 计算
    pow2_tensor = torch.pow(test_tensor, 2)

    # slice reset 计算
    pow2_tensor[2] = 1.

    y = torch.prod(pow2_tensor)
    y.backward()

    print('test_tensor.grad as ', test_tensor.grad)

    ## 根据求梯度的链式法则, 被 slice-reset value 的变量, 由于被赋值了常数, 在梯度反传时它们不再贡献计算 梯度


    ## 注意力机制 Atttention: Q K V --> sum of weights @ V, where weights = f(Q, K)
    ## query/key/value 都有各自的数量和维度. 其中 query 的数量自由决定, 但是其维度要和key相同(毕竟要计算query和key之间的相似度)
    ## value的维度自由决定, 但是其数量要和key相同(key决定了其对应value在最终输出结果中的重要程度)

    ## ScaledDotProductAttention
    # 积式注意力 ScaledDotProductAttention 简单地根据 每条 query和不同keys之间地相似度, 决定了每个key对应的value的权重, 组合出最后的结果
    # 最终由 n_queries 条结果

    class dotProdAttention(torch.nn.Module):

        def __init__(self, dropout):
            super().__init__()
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, Q, K, V, valid_lens=None):
            # Q(batch_size, n_query, qk_size), K(batch_size, n_kv, qk_size), V(batch_size, n_kv, v_size), 
            # valid_lens(batch_size, n_query) or (batch_size)

            # Q K之间 相似度计算
            d = Q.shape[2]
            logits = torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(d) # (batch_size, n_query, n_kv)

            if valid_lens:
                # 如果有 valid_lens, 需要用 mask 确保只有 valid logits 参与生成 概率分布
                from Code.Utils.Common.Mask import valid_slice_mask

                mask = valid_slice_mask(logits.shape, valid_lens)
                logits[~mask] = -1e20 # invalid logits 用 负无穷 slice-reset. 此操作梯度可反传

            # 如果没有 valid_lens, 直接使用 softmax 生成 概率分布 weights
            weights = torch.nn.functional.softmax(logits, dim=-1)

            # 正则化 weights
            return torch.bmm(self.dropout(weights), V)



    Q = torch.tensor([[3, 4, 1, 0],
                      [2, 1, 0, 1],
                      [1, 4, 5, 5]], dtype=torch.float32)
    Q = Q.unsqueeze(0).repeat(3, 1, 1)
    print(Q)

    net = dotProdAttention(0)
    
    y_hat = net(Q, Q, Q)
    print(y_hat)


    ##
    ## 多头注意力:
    ##    单头注意力是指 对 QKV 作各自线性映射(至相同维度 num_hiddens/H )后, 作 ScaledDotProductAttention 后得到 (batch_size, n_queries, num_hiddens/H)
    ## H 个这样的单头注意力的结果, 拼起来是一个 (batch_size, n_queries, num_hiddens) 的结果. 再follow一个 num_hiddens -> num_hiddens 的线性映射

    ## 统一映射到 num_hiddens 维 -> Q(batch_size, n_query, num_hiddens), K(batch_size, n_kv, num_hiddens), V(batch_size, n_kv, num_hiddens)

    ## 若可以切分成 H = num_heads 个头 reshape -> Q(batch_size, H, n_query, w), K(batch_size, H, n_kv, w), V(batch_size, H, n_kv, w)
    ## 那么合并前两个维度, 即得 Q(batch_size*H, n_query, w), K(batch_size*H, n_kv, w), V(batch_size*H, n_kv, w)
    ## 即可完成 DotProdAttention. 

    ## 从 (batch_size, n_, num_hiddens=H*w) 变换到 (batch_size, H, n_, w) 的方法如下:
    ## (batch_size, n_, num_hiddens=H*w) --转置--> (batch_size, num_hiddens=H*w, n_) --> reshape--> (batch_size, H, w, n_) --转置-->  (batch_size, H, n_, w)
    ## --reshape--> (batch_size*H, n_, w)

    def multihead_transpose_qkv(tensor, num_heads):

        batch_size, n_, num_hiddens = tensor.shape # num_hiddens = num_heads * w
        w = num_hiddens // num_heads

        return tensor.permute(0, 2, 1).reshape(batch_size, num_heads, w, n_).permute(0, 1, 3, 2).reshape(-1, n_, w)

    ## Q(batch_size*H, n_query, w) K(batch_size*H, n_kv, w)  V(batch_size*H, n_kv, w)  --dotProdAttention--> (batch_size*H, n_query, w)
    ## 重建过程
    ##  (batch_size*H, n_query, w) --reshape--> (batch_size, H, n_query, w) --转置--> (batch_size, H, w, n_query) --reshape--> (batch_size, H*w, n_query)
    ## --转置--> (batch_size, n_query, num_hiddens=H*w)
    def multihead_transpose_o(tensor, num_heads):
        _, n_query, w = tensor.shape # (batch_size*H, n_query, w)

        return tensor.reshape(-1, num_heads, n_query, w).permute(0, 1, 3, 2).reshape(-1, num_heads*w, n_query).permute(0, 2, 1)



    class MultiHeadAttention(torch.nn.Module):

        def __init__(self, num_heads, num_hiddens, dropout, use_bias=False):
            super().__init__()
            self.H = num_heads
            self.W_q = torch.nn.LazyLinear(num_hiddens, bias=use_bias)
            self.W_k = torch.nn.LazyLinear(num_hiddens, bias=use_bias)
            self.W_v = torch.nn.LazyLinear(num_hiddens, bias=use_bias)
            self.attention = dotProdAttention(dropout)
            self.W_o = torch.nn.LazyLinear(num_hiddens, bias=use_bias)

        def forward(self, Q, K, V, valid_lens=None):
            # Q(batch_size, n_query, qk_size), K(batch_size, n_kv, qk_size), V(batch_size, n_kv, v_size), 
            # valid_lens: (batch_size, n_query) / (batch_size,)  elements are integers <= n_kv

            Q_ = multihead_transpose_qkv( self.W_q(Q), self.H ) # (batch_size*H, n_query, w). w = num_hiddens//H
            K_ = multihead_transpose_qkv( self.W_k(K), self.H ) # (batch_size*H, n_kv, w). w = num_hiddens//H
            # Q_ 和 K_ 确定的 weights: (batch_size*H, n_query, n_kv)
            V_ = multihead_transpose_qkv( self.W_v(V), self.H ) # (batch_size*H, n_kv, w). w = num_hiddens//H

            if valid_lens:
                # valid_lens 需要从 (batch_size, n_query) / (batch_size,) 扩张为(batch_size*H, n_query) / (batch_size*H,)
                valid_lens = torch.repeat_interleave(valid_lens, repeats=self.H, dim=0)

            output = self.attention(Q_, K_, V_, valid_lens) # (batch_size*H, n_query, w). w = num_hiddens//H

            return self.W_o( multihead_transpose_o(output, self.H) ) # (batch_size, n_query, num_hiddens=H*w)

    num_heads, num_hiddens, dropout = 3, 9, 0.
    net = MultiHeadAttention(num_heads, num_hiddens, dropout)
    
    Q = torch.tensor([[3, 4, 1, 0],
                      [2, 1, 0, 1],
                      [1, 4, 5, 5]], dtype=torch.float32)
    Q = Q.unsqueeze(0).repeat(3, 1, 1)
    print(Q)
    
    y_hat = net(Q, Q, Q)
    print(y_hat)