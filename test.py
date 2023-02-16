import os
import warnings
warnings.filterwarnings("ignore")
import torch
from Config import base_data_dir, seq2seq_dir, eng2fra_train_fname, eng2fra_valid_fname, eng2fra_test_fname
from Config import local_model_save_dir
from Code.projs.transformer.Dataset import seq2seqDataset
from Code.projs.transformer.Network import TransformerEncoder, TransformerDecoder, Transformer
from Code.projs.transformer.Trainer import transformerTrainer
from Code.projs.transformer.Evaluator import transformerEpochEvaluator
from Code.Loss.MaskedCELoss import MaskedSoftmaxCELoss
from Code.projs.transformer.Predictor import sentenceTranslator

if __name__ == "__train__":
    # build datasets
    trainset = seq2seqDataset(path=os.path.join(base_data_dir, seq2seq_dir, eng2fra_train_fname), num_steps=8, num_examples=1000)
    validset = seq2seqDataset(path=os.path.join(base_data_dir, seq2seq_dir, eng2fra_valid_fname), num_steps=8)
    testset = seq2seqDataset(path=os.path.join(base_data_dir, seq2seq_dir, eng2fra_test_fname), num_steps=8, num_examples=3)
    # design net & loss
    num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 2, 4, 0.1, False, 4
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    transenc = TransformerEncoder(vocab_size=len(trainset.src_vocab), **test_args)
    transdec = TransformerDecoder(vocab_size=len(trainset.tgt_vocab), **test_args)
    net = Transformer(transenc, transdec)
    loss = MaskedSoftmaxCELoss()
    # init trainer
    trainer = transformerTrainer(net=net, loss=loss, num_epochs=300, batch_size=4)
    ## set the device
    trainer.set_device(torch.device('cuda'))
    ## set the data iters
    trainer.set_data_iter(trainset, validset, testset)
    ## set the optimizer
    trainer.set_optimizer(lr=0.05)
    ## set the grad clipper
    trainer.set_grad_clipping(grad_clip_val=1.0)
    ## set the epoch evaluator
    trainer.set_epoch_eval(transformerEpochEvaluator(num_metrics=2))
    ## set the log file
    trainer.set_log_file('train_logs.txt')
    ## print the lazy topology
    trainer.log_topology('lazy_topo.txt')
    ## check the net & loss
    check_flag = trainer.resolve_net(need_resolve=True)
    if check_flag:
        ## print the defined topology
        trainer.log_topology('def_topo.txt')
        trainer.init_params()
    trainer.fit()
    trainer.save_model('transformer_test.params')

if __name__ == "__main__":
    num_steps = 10
    # set vocabs
    trainset = seq2seqDataset(path=os.path.join(base_data_dir, seq2seq_dir, eng2fra_train_fname), num_steps=num_steps)
    src_vocab, tgt_vocab = trainset.src_vocab, trainset.tgt_vocab
    # set device
    device = torch.device('cpu')
    # load model
    num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 4, 256, 0.2, False, 64
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    transenc = TransformerEncoder(vocab_size=len(trainset.src_vocab), **test_args)
    transdec = TransformerDecoder(vocab_size=len(trainset.tgt_vocab), **test_args)
    net = Transformer(transenc, transdec)
    trained_net_path = os.path.join(local_model_save_dir, 'transformer', 'transformer_v1.params')
    net.load_state_dict(torch.load(trained_net_path, map_location=device))
    # init predictor
    translator = sentenceTranslator(device=device)
    # predict
    src_sentence = 'i\'m home .'
    print(translator.predict(src_sentence, net, src_vocab, tgt_vocab, num_steps=num_steps))
    print('bleu score: ', translator.evaluate())