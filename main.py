import os
import warnings
warnings.filterwarnings("ignore")
import torch
from Config import base_data_dpath, seq2seq_dname, eng2fra_train_fname, eng2fra_valid_fname, eng2fra_test_fname
from Code.projs.transformer.Trainer import transformerTrainer
from Code.projs.transformer.Network import TransformerEncoder, TransformerDecoder, Transformer
from Code.Loss.MaskedCELoss import MaskedSoftmaxCELoss
from Code.projs.transformer.Dataset import seq2seqDataset

if __name__ == "__main__":
    # build datasets
    trainset = seq2seqDataset(path=os.path.join(base_data_dpath, seq2seq_dname, eng2fra_train_fname), num_steps=10)
    validset = seq2seqDataset(path=os.path.join(base_data_dpath, seq2seq_dname, eng2fra_valid_fname), num_steps=10)
    testset = seq2seqDataset(path=os.path.join(base_data_dpath, seq2seq_dname, eng2fra_test_fname), num_steps=10)
    # design net & loss
    num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 4, 256, 0.2, False, 64
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    transenc = TransformerEncoder(vocab_size=len(trainset.src_vocab), **test_args)
    transdec = TransformerDecoder(vocab_size=len(trainset.tgt_vocab), **test_args)
    net = Transformer(transenc, transdec)
    loss = MaskedSoftmaxCELoss()
    # init trainer
    trainer = transformerTrainer(net=net, loss=loss, num_epochs=300, batch_size=128)
    ## set the device
    trainer.set_device(torch.device('cuda'))
    ## set the data iters
    trainer.set_data_iter(trainset, validset, testset)
    ## set the optimizer
    trainer.set_optimizer(lr=0.005)
    ## set the grad clipper
    trainer.set_grad_clipping(grad_clip_val=1.0)
    ## print the lazy topology
    trainer.log_topology('lazy_topo.txt')
    ## check the net & loss
    check_flag = trainer.resolve_net(need_resolve=True)
    if check_flag:
        ## print the defined topology
        trainer.log_topology('def_topo.txt')
        trainer.init_params()
    trainer.fit()
    trainer.save_model('transformer_v1.params')