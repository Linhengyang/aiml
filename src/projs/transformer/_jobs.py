import os
import warnings
warnings.filterwarnings("ignore")
import torch
import typing as t
from .dataset import seq2seqDataset
from .pretrain import transformer_loss
from .trainer import transformerTrainer
from .evaluator import transformerEpochEvaluator
from .predictor import sentenceTranslator
import yaml
from src.utils.text.vocabulize import Vocab
from src.utils.text.glossary import get_BPE_glossary
from src.core.models.transformer import TransformerEncoder, TransformerDecoder, Transformer, transformerConfig

configs = yaml.load(open('src/projs/transformer/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################
# set train log file path / network resolve output path / params save path / source&targe vocabs path

################## glossary and vocabs in workspace/artifact ##################
artifact_dir = os.path.join( configs['artifact_dir'], configs['proj_name'] )

################## params saved in workspace/model ##################
model_dir = os.path.join( configs['model_dir'], configs['proj_name'] )

################## log file in workspace/logs ##################
log_dir = os.path.join( configs['log_dir'], configs['proj_name'] )

################## data-params ##################
context_size = configs['transformerConfig']['max_decoder_ctx_size']

################## train-params ##################
num_epochs, batch_size, lr = 10, 512, 0.00005




# 生产 source & target corpus 的 glossary/vocab
def prepare():
    print('prepare job begin')
    # create all related directories if not existed
    for dir_name in [artifact_dir, model_dir, log_dir]:
        os.makedirs(dir_name, exist_ok=True)
        print(f'directory {dir_name} created')

    # 读取全部语料 corpus
    with open(configs['full_data'], 'r', encoding='utf-8') as f:
        full_data = f.readlines() # full_data 中存在 \t \n 等其他空白符
    corpus = " ".join(full_data)
    uniq_size = len( set(corpus.split(' ')) )
    print(f'unique word size: {uniq_size}')

    # glossary
    glossary_path = os.path.join(artifact_dir, 'glossary.json')
    if not os.path.exists(glossary_path):
        glossary = get_BPE_glossary(corpus, EOW_token = "</w>", merge_times = 50000, save_path = glossary_path)
    else:
        import json
        with open(glossary_path, 'r', encoding='utf-8') as f:
            glossary = json.load(f)
    # vocab
    vocab_path = os.path.join(artifact_dir, 'vocab.json')
    if not os.path.exists(vocab_path):
        vocab = Vocab(corpus, glossary, True, ['<pad>', '<bos>', '<eos>'], '<unk>', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|')
        vocab.save(vocab_path)

    print('prepare job complete')
    return vocab_path


def pretrain():
    print('train job begin')
    # [timetag]
    from datetime import datetime
    now_minute = datetime.now().strftime("%Y-%m-%d_%H:%M")
    # vocab
    vocab_path = os.path.join(artifact_dir, 'vocab.json')

    # /workspace/logs/[proj_name]/train_log_[timetag].txt, defined_net_[timetag].txt
    train_logs_fpath = os.path.join( log_dir, f'train_log_{now_minute}.txt' )
    defined_net_fpath = os.path.join( log_dir, f'defined_net_{now_minute}.txt' )

    # /workspace/model/[proj_name]/saved_params[timetag].params
    saved_params_fpath = os.path.join( model_dir, f'saved_params_{now_minute}.pth' )

    # trainset = seq2seqDataset(configs['train_data'], context_size, vocab_path)
    # validset = seq2seqDataset(configs['valid_data'], context_size, vocab_path)
    testset = seq2seqDataset(configs['test_data'], context_size, vocab_path)

    net_config = transformerConfig(**configs['transformerConfig'])
    net_config.vocab_size = len(testset.vocab)
    transenc = TransformerEncoder(net_config)
    transdec = TransformerDecoder(net_config)
    net = Transformer(transenc, transdec)

    loss = transformer_loss()
    
    # init trainer
    trainer = transformerTrainer(net, loss, num_epochs, batch_size)
    trainer.set_device(torch.device('cuda'))
    trainer.set_data_iter(testset, None, testset)
    trainer.set_optimizer(lr)
    trainer.set_grad_clipping(grad_clip_val=1.0)
    trainer.set_epoch_eval(transformerEpochEvaluator(num_epochs, train_logs_fpath, verbose=True))
    # set trainer
    check_flag = trainer.resolve_net(need_resolve=True, bos_id=testset.vocab['<bos>']) # check the net & loss
    if check_flag:
        trainer.log_topology(defined_net_fpath)
        trainer.init_params()
    
    # fit model
    trainer.fit(bos_id=testset.vocab['<bos>'])
    # save
    trainer.save_model(saved_params_fpath)
    print('train job complete')
    return saved_params_fpath


def translate(saved_params_fpath, vocab_path):
    print('infer job begin')
    # load vocab
    vocab = Vocab()
    vocab.load(vocab_path)
    # set device
    device = torch.device('cuda')

    net_config = transformerConfig(**configs['transformerConfig'])
    net_config.vocab_size = len(vocab)
    transenc = TransformerEncoder(net_config)
    transdec = TransformerDecoder(net_config)
    net = Transformer(transenc, transdec)

    # load params
    net.load_state_dict(torch.load(saved_params_fpath, map_location=device))
    net.eval()
    translator = sentenceTranslator(net, vocab, context_size, configs['temporature'], configs['topk'], device=device)
    
    # predict
    # src_sentence = 'i\'m home .'
    src_sentence = 'Who forced you to do that?'
    print(translator.predict(src_sentence))

    # evaluate output
    # print('bleu score: ', translator.evaluate('je suis chez moi .'))
    print('bleu score: ', translator.evaluate('Qui vous a forcée à faire cela ?'))
    return
