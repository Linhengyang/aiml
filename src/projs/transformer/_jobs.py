import os
import warnings
warnings.filterwarnings("ignore")
import torch
import typing as t
from .dataset import seq2seqDataset
from .network import TransformerEncoder, TransformerDecoder, Transformer, transformer_loss
from .trainer import transformerTrainer
from .evaluator import transformerEpochEvaluator
from .predictor import sentenceTranslator
import yaml
from src.utils.text.vocabulize import Vocab
from src.utils.text.glossary import get_BPE_glossary

configs = yaml.load(open('src/projs/transformer/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################
# set train log file path / network resolve output path / params save path / source&targe vocabs path

################## symbols and vocabs in workspace/cache ##################
glossary_dir = os.path.join( configs['artifact_dir'], configs['proj_name'], 'glossary' )
vocab_dir = os.path.join( configs['artifact_dir'], configs['proj_name'], 'vocab' )



################## params saved in workspace/model ##################
model_dir = os.path.join( configs['model_dir'], configs['proj_name'] )



################## log file in workspace/logs ##################
log_dir = os.path.join( configs['log_dir'], configs['proj_name'] )



################## data-params ##################
num_steps = configs['num_steps']



################## network-params ##################
num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 4, 256, 0.1, False, 64



################## train-params ##################
num_epochs, batch_size, lr = 5, 512, 0.00005



# num_blk, num_heads, num_hiddens, dropout, use_bias, ffn_num_hiddens = 2, 4, 256, 0.1, False, 64
# batch_size
# ---------------------> 4GB 显存



# 生产 source corpus 和 target corpus 的 glossary/vocab
def prepare_job():
    print('prepare job begin')

    # create all related directories if not existed
    for dir_name in [glossary_dir, vocab_dir, model_dir, log_dir]:
        os.makedirs(dir_name, exist_ok=True)
        print(f'directory {dir_name} created')

    # generate glossary of source/target language and save them

    # 读取全部语料 corpus
    with open(configs['full_data'], 'r', encoding='utf-8') as f:
        full_data = f.readlines() # full_data 中存在 \t \n 等其他空白符

    # 分割成 english 和 france. 
    # 由于本次任务对 source / target language 采用 分开独立的词汇表 和 嵌入空间, 故 BPE 生成 glossary 也是独立的
    eng_corpus, fra_corpus = [], []
    for line in full_data:
        eng_sentence, fra_sentence = line.split('\t')
        eng_corpus.append(eng_sentence)
        fra_corpus.append(fra_sentence)
    
    eng_corpus = " ".join(eng_corpus) # concate 所有行
    fra_corpus = " ".join(fra_corpus) # concate 所有行

    eng_uniq_size = len( set(eng_corpus.split(' ')) )
    fra_uniq_size = len( set(fra_corpus.split(' ')) )
    print(f'english unique word size:{eng_uniq_size} \n france unique word size:{fra_uniq_size}')


    # glossary
    # 当 eng_glossary_path 文件不存在时, 生产并保存 eng_glossary 到 glossary_dir/english.json
    eng_glossary_path = os.path.join(glossary_dir, 'english.json')
    if not os.path.exists(eng_glossary_path):
        # create english glossary
        eng_glossary = get_BPE_glossary(corpus = eng_corpus, EOW_token = "</w>", merge_times = 15000, save_path = eng_glossary_path)
    

    # 当 fra_glossary_path 文件不存在时, 生产并保存 fra_glossary 到 glossary_dir/france.json
    fra_glossary_path = os.path.join(glossary_dir, 'france.json')
    if not os.path.exists(fra_glossary_path):
        # create france glossary
        fra_glossary = get_BPE_glossary(corpus = fra_corpus, EOW_token = "</w>", merge_times = 25000, save_path = fra_glossary_path)


    # vocab
    # 当 eng_vocab_path 文件不存在时, 生产并保存 eng_vocab 到 vocab_dir/english.json
    eng_vocab_path = os.path.join(vocab_dir, 'english.json')

    if not os.path.exists(eng_vocab_path):
        eng_vocab = Vocab(corpus=eng_corpus, glossary=eng_glossary, need_lower=True, reserved_tokens=['<pad>', '<bos>', '<eos>'],
                          unk_token='<unk>', separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|')
        eng_vocab.save(eng_vocab_path)
    

    # 当 fra_vocab_path 文件不存在时, 生产并保存 fra_vocab 到 vocab_dir/france.json
    fra_vocab_path = os.path.join(vocab_dir, 'france.json')

    if not os.path.exists(fra_vocab_path):
        fra_vocab = Vocab(corpus=fra_corpus, glossary=fra_glossary, need_lower=True, reserved_tokens=['<pad>', '<bos>', '<eos>'],
                          unk_token='<unk>', separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|')
        fra_vocab.save(fra_vocab_path)


    print('prepare job complete')
    return eng_vocab_path, fra_vocab_path


def train_job(eng_vocab_path, fra_vocab_path):
    print('train job begin')
    # [timetag]
    from datetime import datetime
    now_minute = datetime.now().strftime("%Y-%m-%d_%H:%M")

    # /workspace/logs/[proj_name]/train_log_[timetag].txt, defined_net_[timetag].txt
    train_logs_fpath = os.path.join( log_dir, f'train_log_{now_minute}.txt' )
    defined_net_fpath = os.path.join( log_dir, f'defined_net_{now_minute}.txt' )

    # /workspace/model/[proj_name]/saved_params[timetag].params
    saved_params_fpath = os.path.join( model_dir, f'saved_params_{now_minute}.pth' )

    trainset = seq2seqDataset(text_path=configs['train_data'], num_steps=num_steps,
                              src_vocab_path=eng_vocab_path, tgt_vocab_path=fra_vocab_path)
    validset = seq2seqDataset(text_path=configs['valid_data'], num_steps=num_steps,
                              src_vocab_path=eng_vocab_path, tgt_vocab_path=fra_vocab_path)
    testset = seq2seqDataset(text_path=configs['test_data'], num_steps=num_steps,
                             src_vocab_path=eng_vocab_path, tgt_vocab_path=fra_vocab_path)

    # design net & loss
    net_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    
    src_vocab_size, tgt_vocab_size = len(trainset.src_vocab), len(trainset.tgt_vocab)
    transenc = TransformerEncoder(vocab_size=src_vocab_size, **net_args)
    transdec = TransformerDecoder(vocab_size=tgt_vocab_size, **net_args)

    net = Transformer(transenc, transdec)
    loss = transformer_loss()

    # init trainer
    trainer = transformerTrainer(net, loss, num_epochs, batch_size)

    trainer.set_device(torch.device('cuda')) # set the device
    trainer.set_data_iter(trainset, validset, testset) # set the data iters
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

    print('train job complete')
    return saved_params_fpath


def infer_job(saved_params_fpath, eng_vocab_path, fra_vocab_path):
    print('infer job begin')

    # load vocabs
    src_vocab, tgt_vocab = Vocab(), Vocab()

    # source and target language vocabs
    src_vocab.load( eng_vocab_path )
    tgt_vocab.load( fra_vocab_path )

    # set device
    device = torch.device('cuda')

    # construct model
    net_args = {"num_heads":num_heads, "num_hiddens":num_hiddens, "dropout":dropout,
                "use_bias":use_bias, "ffn_num_hiddens":ffn_num_hiddens, "num_blk":num_blk}
    
    transenc = TransformerEncoder(vocab_size=len(src_vocab), **net_args)
    transdec = TransformerDecoder(vocab_size=len(tgt_vocab), **net_args)
    net = Transformer(transenc, transdec).to(device)

    # load params
    net.load_state_dict(torch.load(saved_params_fpath, map_location=device))
    net.eval()

    # set predictor
    search_mode = 'beam'
    # if dropout > 0.0: # 当网络有随机性时, 必须用贪心搜索预测. 因为束搜索要用 train mode 网络. 而 train mode 的网络在每次推理时会产生随机性
    #     search_mode = 'greedy'
    
    translator = sentenceTranslator(src_vocab, tgt_vocab, net, num_steps, search_mode,
                                    device=device, beam_size=3, length_factor=5) #加大对长句的奖励

    # predict
    # src_sentence = 'i\'m home .'
    src_sentence = 'Who forced you to do that?'
    print(translator.predict(src_sentence))

    # evaluate output
    # print('bleu score: ', translator.evaluate('je suis chez moi .'))
    print('bleu score: ', translator.evaluate('Qui vous a forcée à faire cela ?'))
    print('pred score: ', translator.pred_scores)

    print('infer job complete')
    return
