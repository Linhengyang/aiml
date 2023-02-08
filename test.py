import os
import torch
from Config import base_data_dpath, seq2seq_dname, eng2fra_fname
# import Code.projs.transformer.DataLoad_seq2seq as test1
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
    ## 测试train
    batch_size, num_steps, d_dim = 2, 3, 6
    test_input_shape = (batch_size, num_steps, d_dim)
    test_src_X = torch.ones(test_input_shape)
    test_src_valid_lens = torch.tensor([1, 2])
    test_tgt_X = torch.ones(test_input_shape)
    # 设定模型参数
    num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 6, 0.1, False, 8
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens}
    # encode
    test_enc = test2.TransformerEncoderBlock(**test_args)
    enc_info = (test_enc(test_src_X, test_src_valid_lens), test_src_valid_lens)
    # decode
    test_dec = test2.TransformerDecoderBlock(blk_ind=0, **test_args)
    dec_output = test_dec(test_tgt_X, enc_info)
    print(dec_output[0].shape)
    print(dec_output[1])

    # ## 测试infer
    # batch_size, num_steps, d_dim = 1, 3, 6
    # test_input_shape = (batch_size, num_steps, d_dim)
    # test_src_X = torch.ones(test_input_shape)
    # test_src_valid_lens = torch.tensor([num_steps,])
    # test_tgt_X = torch.ones((1, 1, 6))
    # # 设定模型参数
    # num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 6, 0.1, False, 8
    # test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
    #              "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens}
    # # encode
    # test_enc = test2.TransformerEncoderBlock(**test_args)
    # enc_info = (test_enc(test_src_X, test_src_valid_lens), test_src_valid_lens)
    # # decode
    # test_dec = test2.TransformerDecoderBlock(blk_ind=0, **test_args)
    # test_dec.eval()
    # dec_output = test_dec(test_tgt_X, enc_info, {})
    # print(dec_output[0].shape)
    # print(dec_output[1])