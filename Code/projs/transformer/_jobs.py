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
from ...Utils.Text.BytePairEncoding import get_BPE_symbols, segment_word_BPE_greedy

configs = yaml.load(open('Code/projs/transformer/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################
symbols_dir = os.path.join( configs['cache_dir'], configs['proj_name'], 'symbols' )
vocabs_dir = os.path.join( configs['cache_dir'], configs['proj_name'], 'vocabs' )
model_proj_dir = os.path.join( configs['model_dir'], configs['proj_name'] )
# set train log file path / network resolve output path / params save path / source&targe vocabs path
log_proj_dir = os.path.join( configs['log_dir'], configs['proj_name'] )
# directories for vocabs
src_vocab_dir = os.path.join(vocabs_dir, 'source')
tgt_vocab_dir = os.path.join(vocabs_dir, 'target')




################## data-params ##################
num_steps = configs['num_steps']







################## network-params ##################
num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 3, 256, 0.1, False, 64










################## train-params ##################
num_epochs, batch_size, lr = 100, 512, 0.0005




def prepare_job():

    import json
    # generate symbols of source/target language and save them

    # 读取全部语料 corpus
    with open(configs['full_data'], 'r', encoding='utf-8') as f:
        full_data = f.readlines() # full_data 中存在 \t \n 等其他空白符

    # 分割成 english 和 france. 
    # 由于本次任务对 source language 和 target language 采用 分开独立的词汇表 和 嵌入空间, 故 BPE 生成 symbols 也是独立的
    eng_corpus, fra_corpus = [], []

    for line in full_data:
        eng_sentence, fra_sentence = line.split('\t')
        eng_corpus.append(eng_sentence)
        fra_corpus.append(fra_sentence)
    
    eng_corpus = " ".join(eng_corpus)
    fra_corpus = " ".join(fra_corpus)

    # create symbols
    eng_symbols = get_BPE_symbols(
        text = eng_corpus,
        tail_token = "</w>",
        merge_times = 8000,
        need_lower = True,
        separate_puncs = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
        normalize_whitespace = True,
        )

    # 当 symbols 文件不存在时, 保存 symbols 到 symbols_dir/source.json
    eng_symbols_path = os.path.join(symbols_dir, 'source.json')
    if not os.path.exists(eng_symbols_path):
        with open(eng_symbols_path, 'w') as f:
            json.dump(eng_symbols, f)

    fra_symbols = get_BPE_symbols(
        text = fra_corpus,
        tail_token = "</w>",
        merge_times = 8000,
        need_lower = True,
        separate_puncs = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
        normalize_whitespace = True,
        )

    # 当 eng_symbols_path 文件不存在时, 保存 symbols 到 symbols_dir/target.json
    fra_symbols_path = os.path.join(symbols_dir, 'target.json')
    if not os.path.exists(fra_symbols_path):
        with open(fra_symbols_path, 'w') as f:
            json.dump(fra_symbols, f)

    return len(eng_symbols), len(fra_symbols)



def train_job():

    # [timetag]
    from datetime import datetime
    now_minute = datetime.now().strftime("%Y-%m-%d_%H:%M")
    # /workspace/logs/[proj_name]/train_log_[timetag].txt, defined_net_[timetag].txt
    train_logs_fpath = os.path.join( log_proj_dir, f'train_log_{now_minute}.txt' )
    defined_net_fpath = os.path.join( log_proj_dir, f'defined_net_{now_minute}.txt' )

    # /workspace/model/[proj_name]/saved_params[timetag].params
    saved_params_fpath = os.path.join( model_proj_dir, f'saved_params_{now_minute}.txt' )

    # build datasets
    full_dataset = seq2seqDataset(configs['full_data'], num_steps=num_steps)

    # save source and target vocabs
    src_vocab = full_dataset.src_vocab
    tgt_vocab = full_dataset.tgt_vocab

    # validset & testset
    valid_dataset = seq2seqDataset(configs['valid_data'], num_steps=num_steps)
    test_dataset = seq2seqDataset(configs['test_data'], num_steps=num_steps)
    
    # /workspace/cache/[proj_name]/vocabs/source/idx2token.json, token2idx.json
    src_vocab.save( src_vocab_dir )
    # /workspace/cache/[proj_name]/vocabs/target/idx2token.json, token2idx.json
    tgt_vocab.save( tgt_vocab_dir )

    # design net & loss
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    
    transenc = TransformerEncoder(vocab_size=len(src_vocab), **test_args)
    transdec = TransformerDecoder(vocab_size=len(tgt_vocab), **test_args)

    net = Transformer(transenc, transdec)
    loss = MaskedSoftmaxCELoss()

    # init trainer
    trainer = transformerTrainer(net, loss, num_epochs, batch_size)

    trainer.set_device(torch.device('cuda')) # set the device
    trainer.set_data_iter(full_dataset, valid_dataset, test_dataset) # set the data iters
    trainer.set_optimizer(lr) # set the optimizer
    trainer.set_grad_clipping(grad_clip_val=1.0) # set the grad clipper
    trainer.set_epoch_eval(transformerEpochEvaluator(num_epochs, train_logs_fpath, verbose=True)) # set the epoch evaluator

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
    num_steps = num_steps

    # load vocabs
    from ....Code.Utils.Text.Vocabulize import Vocab
    src_vocab, tgt_vocab = Vocab(), Vocab()

    # /workspace/cache/[proj_name]/vocabs/source/idx2token.json, token2idx.json
    src_vocab.load( src_vocab_dir )
    # /workspace/cache/[proj_name]/vocabs/target/idx2token.json, token2idx.json
    tgt_vocab.load( tgt_vocab_dir )

    # set device
    device = torch.device('cpu')

    # construct model
    test_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                 "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    
    transenc = TransformerEncoder(vocab_size=len(src_vocab), **test_args)
    transdec = TransformerDecoder(vocab_size=len(tgt_vocab), **test_args)
    net = Transformer(transenc, transdec)

    # load params
    latest_trained_net_path = max( os.listdir( model_proj_dir ) )
    net.load_state_dict(torch.load(latest_trained_net_path, map_location=device))
    net.eval()

    # set predictor
    search_mode = 'beam'
    # if dropout > 0.0: # 当网络有随机性时, 必须用贪心搜索预测. 因为束搜索要用 train mode 网络. 而 train mode 的网络在每次推理时会产生随机性
    #     search_mode = 'greedy'
    
    translator = sentenceTranslator(search_mode=search_mode, device=device, length_factor=5) #加大对长句的奖励

    # predict
    src_sentence = 'i\'m home .'
    print(translator.predict(src_sentence, net, src_vocab, tgt_vocab, num_steps=num_steps))

    # evaluate output
    print('bleu score: ', translator.evaluate('je suis chez moi .'))
    print('pred score: ', translator.pred_scores)
