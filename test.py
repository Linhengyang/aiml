import os
import warnings
warnings.filterwarnings("ignore")
import torch
from Config import base_data_dpath, seq2seq_dname, eng2fra_fname
from Code.projs.transformer.DataLoad_seq2seq import data_loader_seq2seq
from Code.projs.transformer.Trainer import transformerTrainer
from Code.projs.transformer.Network import TransformerEncoder, TransformerDecoder, Transformer

if __name__ == "__main__":
    eng2fra = os.path.join(base_data_dpath, seq2seq_dname, eng2fra_fname)
    train_iter, src_vocab, tgt_vocab = data_loader_seq2seq(path=eng2fra, batch_size=2, num_steps=8, num_examples=600)

    num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 2, 6, 0.1, False, 8
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    transenc = TransformerEncoder(vocab_size=len(src_vocab), **test_args)
    transdec = TransformerDecoder(vocab_size=len(tgt_vocab), **test_args)
    net = Transformer(transenc, transdec)

    trainer = transformerTrainer(net, None, train_iter, None, None, None)
    trainer.topo_logger('lazy_topo.txt')
    trainer.net_resolver()
    trainer.topo_logger('defined_topo.txt')