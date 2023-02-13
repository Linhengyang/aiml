import os
import warnings
warnings.filterwarnings("ignore")
import torch
from Config import base_data_dpath, seq2seq_dname, eng2fra_fname
import Code.projs.transformer.DataLoad_seq2seq as test1
# import Code.Modules._transformer as test2
import Code.projs.transformer.Network as test3
from Code.Loss.MaskedCELoss import MaskedSoftmaxCELoss
## net tester
# 拿到类定义, 手动net_args定义, 手动input_batch_shape定义
# 如果没有data loader, 自动生成 input batch;如果有data loader, iter for 1 input batch, 自动init一个实例net, 自动前馈net(inputs), 得到outputs
# 检测output是否满足条件

## dataloader tester
# 拿到函数定义, 手动input args定义
# 自动call(input args), 得到data iter. iter 1次, 得到minibatch
# 检测batch是否满足条件

if __name__ == "__main__":
    eng2fra = os.path.join(base_data_dpath, seq2seq_dname, eng2fra_fname)
    train_iter, src_vocab, tgt_vocab = test1.data_loader_seq2seq(path=eng2fra, batch_size=2, num_steps=8, num_examples=600)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X)
        print('valid lengths for X:', X_valid_len)
        print('Y:', Y)
        print('valid lengths for Y:', Y_valid_len)
        print(X.dtype, Y.dtype)
        print("**********")
        break
    num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 2, 6, 0.1, False, 8
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    transenc = test3.TransformerEncoder(vocab_size=len(src_vocab), **test_args)
    transdec = test3.TransformerDecoder(vocab_size=len(tgt_vocab), **test_args)
    net = test3.Transformer(transenc, transdec)
    print("********** net before first batch **********")
    # print(net)
    print('********* test train transformer **********')
    net.train()
    dec_outputs = net(X, Y, X_valid_len)
    print(dec_outputs[0].shape == torch.Size([2, 8, len(tgt_vocab)]))
    print(dec_outputs[1] is None)
    print("********** net after first batch **********")
    # print(net)
    print("********** test loss **********")
    loss = MaskedSoftmaxCELoss()
    l = loss(dec_outputs[0], Y, Y_valid_len)
    print(' loss is ', l)
    print('Y shape: ', Y.shape)
    print('Y_valid_lens shape: ', Y_valid_len.shape)
    print("********** test BP **********")
    l.sum().backward()
    # print('********* test infer transformer **********')
    # net.eval()
    # eng = "i lost ."
    # src_X = torch.tensor(src_vocab[eng.split()] + [src_vocab['<eos>']]).unsqueeze(0)
    # print(eng, " now is ", src_X)
    # src_valid_len = torch.tensor([src_X.shape[1], ])
    # print(eng, " valid length is ", src_valid_len)
    # enc_outputs = net.encoder(src_X, src_valid_len)

    # enc_info = net.decoder.init_state(enc_outputs)
    # infer_recorder = {}
    # num_preds = 3
    # tgt_X = torch.tensor( [tgt_vocab['<bos>'],] ).unsqueeze(0)
    # print('initial tgt token is ', tgt_X)
    # print('initial infer recorder is ', infer_recorder)
    # for i in range(num_preds):
    #     X_hat, infer_recorder = net.decoder(tgt_X, enc_info, infer_recorder)
    #     print('infer recorder at step ', i, ' is ', infer_recorder)
    #     print('X_hat shape is ', X_hat.shape)
    #     tgt_X = X_hat.argmax(dim=-1)
    #     print('tgt token at step ', i, ' is ', tgt_vocab.to_tokens(tgt_X.item()))