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

vocabs_dir = os.path.join( [configs['cache_dir'], configs['proj_name'], '/vocabs'] )
model_proj_dir = os.path.join( [configs['model_dir'], configs['proj_name']] )


def train_job():
    # set train log file path / network resolve output path / params save path / source&targe vocabs path
    log_proj_dir = os.path.join( [configs['log_dir'], configs['proj_name']] )

    # [timetag]
    from datetime import datetime
    now_minute = datetime.now().strftime("%Y-%m-%d_%H:%M")
    # /workspace/logs/[proj_name]/train_log_[timetag].txt, defined_net_[timetag].txt
    train_logs_fpath = os.path.join( [log_proj_dir, f'train_log_{now_minute}.txt'] )
    defined_net_fpath = os.path.join( [log_proj_dir, f'defined_net_{now_minute}.txt'] )

    # /workspace/model/[proj_name]/saved_params[timetag].params
    saved_params_fpath = os.path.join( [model_proj_dir, f'saved_params_{now_minute}.txt'] )

    # build datasets
    trainset = seq2seqDataset(configs['train_data'], num_steps=10)

    # save source and target vocabs
    src_vocab = trainset.src_vocab
    tgt_vocab = trainset.tgt_vocab

    # /workspace/cache/[proj_name]/vocabs/source/idx2token.json, token2idx.json
    src_vocab.save( os.path.join([vocabs_dir, '/source']) )

    # /workspace/cache/[proj_name]/vocabs/target/idx2token.json, token2idx.json
    tgt_vocab.save( os.path.join([vocabs_dir, '/target']) )

    # design net & loss
    num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 4, 256, 0.2, False, 64
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    
    transenc = TransformerEncoder(vocab_size=len(trainset.src_vocab), **test_args)
    transdec = TransformerDecoder(vocab_size=len(trainset.tgt_vocab), **test_args)

    net = Transformer(transenc, transdec)
    loss = MaskedSoftmaxCELoss()

    # init trainer
    num_epochs, batch_size, lr = 100, 128, 0.0005
    trainer = transformerTrainer(net, loss, num_epochs, batch_size)

    trainer.set_device(torch.device('cuda')) # set the device
    trainer.set_data_iter(trainset, None, None) # set the data iters
    trainer.set_optimizer(lr) # set the optimizer
    trainer.set_grad_clipping(grad_clip_val=1.0) # set the grad clipper
    trainer.set_epoch_eval(transformerEpochEvaluator(num_epochs, train_logs_fpath, verbose=False)) # set the epoch evaluator

    # set trainer
    check_flag = trainer.resolve_net(need_resolve=True)## check the net & loss
    if check_flag:
        trainer.log_topology(defined_net_fpath)## print the defined topology
        trainer.init_params()## init params
    
    # fit model
    trainer.fit()

    # save
    trainer.save_model(saved_params_fpath)








def infer_job():
    num_steps = 10

    # load vocabs
    from ....Code.Utils.Text.Vocabulize import Vocab
    src_vocab, tgt_vocab = Vocab(), Vocab()

    # /workspace/cache/[proj_name]/vocabs/source/idx2token.json, token2idx.json
    src_vocab.load( os.path.join([vocabs_dir, '/source']) )

    # /workspace/cache/[proj_name]/vocabs/target/idx2token.json, token2idx.json
    tgt_vocab.load( os.path.join([vocabs_dir, '/target']) )

    # set device
    device = torch.device('cpu')

    # construct model
    num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 4, 256, 0.1, False, 64
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    
    transenc = TransformerEncoder(vocab_size=len(src_vocab), **test_args)
    transdec = TransformerDecoder(vocab_size=len(tgt_vocab), **test_args)
    net = Transformer(transenc, transdec)


    # load params
    latest_trained_net_path = max( os.listdir( model_proj_dir ) )
    net.load_state_dict(torch.load(latest_trained_net_path, map_location=device))

    # set predictor
    search_mode = 'beam'
    if dropout > 0.0: # 当网络有随机性时, 必须用贪心搜索预测. 因为束搜索要用 train mode 网络. 而 train mode 的网络在每次推理时会产生随机性
        search_mode = 'greedy'
    
    translator = sentenceTranslator(search_mode=search_mode, device=device, alpha=5) #加大对长句的奖励

    # predict
    src_sentence = 'i\'m home .'
    print(translator.predict(src_sentence, net, src_vocab, tgt_vocab, num_steps=num_steps))

    # evaluate output
    print('bleu score: ', translator.evaluate('je suis chez moi .'))
    print('pred score: ', translator.pred_scores)
