import os
import warnings
warnings.filterwarnings("ignore")
import torch
from Config import base_data_dpath, seq2seq_dname, eng2fra_fname
from Code.projs.transformer.DataLoad_seq2seq import data_loader_seq2seq
from Code.projs.transformer.Network import TransformerEncoder, TransformerDecoder, Transformer

if __name__ == "__main__":
    eng2fra = os.path.join(base_data_dpath, seq2seq_dname, eng2fra_fname)
    train_iter, src_vocab, tgt_vocab = data_loader_seq2seq(path=eng2fra, batch_size=2, num_steps=8, num_examples=600)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('first batch generated')
        break
    num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 2, 6, 0.1, False, 8
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    transenc = TransformerEncoder(vocab_size=len(src_vocab), **test_args)
    transdec = TransformerDecoder(vocab_size=len(tgt_vocab), **test_args)
    net = Transformer(transenc, transdec)
    print("********** net before first batch **********")
    print(net)
    
    net.train()
    dec_outputs = net(X, Y, X_valid_len)
    print(dec_outputs[0].shape == torch.Size([2, 8, len(tgt_vocab)]))
    print(dec_outputs[1] is None)
    print("********** net after first batch **********")
    print(net)
