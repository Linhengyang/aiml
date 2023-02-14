import os
import warnings
warnings.filterwarnings("ignore")
import torch
from Config import base_data_dpath, seq2seq_dname, eng2fra_train_fname, eng2fra_valid_fname, eng2fra_test_fname
from Code.projs.transformer.DataLoad_seq2seq import data_loader_seq2seq
from Code.projs.transformer.Trainer import transformerTrainer
from Code.projs.transformer.Network import TransformerEncoder, TransformerDecoder, Transformer
from Code.Loss.MaskedCELoss import MaskedSoftmaxCELoss

if __name__ == "__main__":
    eng2fra_train = os.path.join(base_data_dpath, seq2seq_dname, eng2fra_train_fname)
    eng2fra_valid = os.path.join(base_data_dpath, seq2seq_dname, eng2fra_valid_fname)
    eng2fra_test = os.path.join(base_data_dpath, seq2seq_dname, eng2fra_test_fname)
    train_iter, src_vocab, tgt_vocab = data_loader_seq2seq(path=eng2fra_train, batch_size=2, num_steps=8, num_examples=600)
    valid_iter, _, _ = data_loader_seq2seq(path=eng2fra_valid, batch_size=2, num_steps=8)
    test_iter, _, _ = data_loader_seq2seq(path=eng2fra_test, batch_size=2, num_steps=8, num_examples=3)
    # design net & loss
    num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 2, 6, 0.1, False, 8
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    transenc = TransformerEncoder(vocab_size=len(src_vocab), **test_args)
    transdec = TransformerDecoder(vocab_size=len(tgt_vocab), **test_args)
    net = Transformer(transenc, transdec)
    loss = MaskedSoftmaxCELoss()
    # init trainer
    trainer = transformerTrainer(net=net, loss=loss, num_epochs=3)
    ## set the device
    trainer.set_device(torch.device('cuda'))
    ## set the data iters
    trainer.set_data_iter(tgt_vocab, train_iter, valid_iter, test_iter)
    ## set the optimizer
    trainer.set_optimizer(lr=0.005)
    ## set the grad clipper
    trainer.set_grad_clipping(grad_clip_val=None)
    ## print the lazy topology
    trainer.log_topology('lazy_topo.txt')
    ## check the net & loss
    check_flag = trainer.resolve_net(need_resolve=True)
    if check_flag:
        ## print the defined topology
        trainer.log_topology('def_topo.txt')
        trainer.init_params()
    # trainer.fit()