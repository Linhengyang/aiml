import os
import torch
from Config import base_data_dpath, seq2seq_dname, eng2fra_fname
import Code.projs.transformer.DataLoad_seq2seq as test1
import Code.Modules._transformer as test2

## net tester
# 拿到类定义, 手动net_args定义, 手动input_batch_shape定义
# 如果没有data loader, 自动生成 input batch;如果有data loader, iter for 1 input batch, 自动init一个实例net, 自动前馈net(inputs), 得到outputs
# 检测output是否满足条件

## dataloader tester
# 拿到函数定义, 手动input args定义
# 自动call(input args), 得到data iter. iter 1次, 得到minibatch
# 检测batch是否满足条件

if __name__ == "__main__":
    ### test1
    # eng2fra = os.path.join(base_data_dpath, seq2seq_dname, eng2fra_fname)
    # train_iter, src_vocab, tgt_vocab = test1.data_loader_seq2seq(path=eng2fra, batch_size=2, num_steps=8, num_examples=600)
    # for X, X_valid_len, Y, Y_valid_len in train_iter:
    #     print('X:', X)
    #     print('valid lengths for X:', X_valid_len)
    #     print('Y:', Y)
    #     print('valid lengths for Y:', Y_valid_len)
    #     print(X.dtype, Y.dtype)
    #     break
    ### test2
    # batch_size, num_steps, d_dim = 2, 3, 4
    # test_input_shape = (batch_size, num_steps, d_dim)
    # test_input = torch.ones(test_input_shape)
    # test_args = (4, 8)
    # testnet = test2.PositionWiseFFN(*test_args)
    # testnet.eval()
    # print(testnet(test_input).shape)
    