import os
import warnings
warnings.filterwarnings("ignore")
import torch
from .Dataset import seq2seqDataset
from .Network import TransformerEncoder, TransformerDecoder, Transformer
from ...Loss.MaskedCELoss import MaskedSoftmaxCELoss
from .Trainer import transformerTrainer
from .Evaluator import transformerEpochEvaluator
from .Predictor import sentenceTranslator
import yaml
configs = yaml.load(open('Code/projs/transformer/configs.yaml', 'rb'), Loader=yaml.FullLoader)
base_data_dir, seq2seq_dir = configs['base_data_dir'], configs['seq2seq_dir']
eng2fra_train_fname,eng2fra_valid_fname, eng2fra_test_fname = configs['eng2fra_train_fname'], configs['eng2fra_valid_fname'], configs['eng2fra_test_fname']
local_model_save_dir = configs['local_model_save_dir']


def train_job():
    # build datasets
    trainset = seq2seqDataset(path=os.path.join(base_data_dir, seq2seq_dir, eng2fra_train_fname), num_steps=10, num_examples=4000)
    validset = seq2seqDataset(path=os.path.join(base_data_dir, seq2seq_dir, eng2fra_valid_fname), num_steps=10)
    testset = seq2seqDataset(path=os.path.join(base_data_dir, seq2seq_dir, eng2fra_test_fname), num_steps=10)
    # design net & loss
    num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 2, 4, 0.1, False, 4
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    transenc = TransformerEncoder(vocab_size=len(trainset.src_vocab), **test_args)
    transdec = TransformerDecoder(vocab_size=len(trainset.tgt_vocab), **test_args)
    net = Transformer(transenc, transdec)
    loss = MaskedSoftmaxCELoss()
    # init trainer
    trainer = transformerTrainer(net=net, loss=loss, num_epochs=100, batch_size=128)
    trainer.set_device(torch.device('cuda'))## set the device
    trainer.set_data_iter(trainset, validset, testset)## set the data iters
    trainer.set_optimizer(lr=0.0005)## set the optimizer
    trainer.set_grad_clipping(grad_clip_val=1.0)## set the grad clipper
    trainer.set_epoch_eval(transformerEpochEvaluator('train_logs.txt', visualizer=True))## set the epoch evaluator
    # start
    trainer.log_topology('lazy_topo.txt')## print the lazy topology
    check_flag = trainer.resolve_net(need_resolve=True)## check the net & loss
    if check_flag:
        trainer.log_topology('def_topo.txt')## print the defined topology
        trainer.init_params()## init params
    # fit
    trainer.fit()
    # save
    trainer.save_model('transformer_test.params')

def infer_job():
    num_steps = 10
    # set vocabs
    trainset = seq2seqDataset(path=os.path.join(base_data_dir, seq2seq_dir, eng2fra_train_fname), num_steps=num_steps)
    src_vocab, tgt_vocab = trainset.src_vocab, trainset.tgt_vocab
    device = torch.device('cpu')# set device
    # load model
    num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 4, 256, 0.0, False, 64
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    transenc = TransformerEncoder(vocab_size=len(trainset.src_vocab), **test_args)
    transdec = TransformerDecoder(vocab_size=len(trainset.tgt_vocab), **test_args)
    net = Transformer(transenc, transdec)
    trained_net_path = os.path.join(local_model_save_dir, 'transformer', 'transformer_v1.params')
    net.load_state_dict(torch.load(trained_net_path, map_location=device))
    # init predictor
    search_mode = 'beam'
    if dropout > 0.0: # 当网络有随机性时, 必须用贪心搜索预测. 因为束搜索要用train mode网络
        search_mode = 'greedy'
    translator = sentenceTranslator(search_mode=search_mode, device=device)
    # predict
    src_sentence = 'i\'m home .'
    print(translator.predict(src_sentence, net, src_vocab, tgt_vocab, num_steps=num_steps))
    # evaluate
    print('bleu score: ', translator.evaluate())
    print('pred score: ', translator.pred_scores)