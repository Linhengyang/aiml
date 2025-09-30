import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from .dataset import FMNISTDatasetOnline, FMNISTDatasetLocal
from .network import ViTEncoder
from .trainer import vitTrainer
from .evaluator import vitEpochEvaluator
from .predictor import fmnistClassifier
import yaml
import json

configs = yaml.load(open('src/projs/vit/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################## directories ##################
# set train log file path / network resolve output path / params save path / source&targe vocabs path


################## image arguments in workspace/cache ##################
# json file for image data arguments
cache_dir = os.path.join( configs['cache_dir'], configs['proj_name'] )
image_args = os.path.join( cache_dir, 'image_args.json' )



################## params saved in workspace/model ##################
model_dir = os.path.join( configs['model_dir'], configs['proj_name'] )



################## log file in workspace/logs ##################
log_dir = os.path.join( configs['log_dir'], configs['proj_name'] )










################## data-params ##################
patch_size = (7, 7)



################## network-params ##################
num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens = 2, 4, 64, 0.1, 0.1, 128





################## train-params ##################
num_epochs, batch_size, lr = 2, 128, 0.0005








def prepare_job():
    print('prepare job begin')

    # create all related directories if not existed
    for dir_name in [cache_dir, model_dir, log_dir]:
        os.makedirs(dir_name, exist_ok=True)
        print(f'directory {dir_name} created')

    print('prepare job complete')





def train_job(data_source):
    print('train job begin')
    # [timetag]
    from datetime import datetime
    now_minute = datetime.now().strftime("%Y-%m-%d_%H:%M")

    # /workspace/logs/[proj_name]/train_log_[timetag].txt, defined_net_[timetag].txt
    train_logs_fpath = os.path.join( log_dir, f'train_log_{now_minute}.txt' )
    defined_net_fpath = os.path.join( log_dir, f'defined_net_{now_minute}.txt' )

    # /workspace/model/[proj_name]/saved_params[timetag].params
    saved_params_fpath = os.path.join( model_dir, f'saved_params_{now_minute}.pth' )

    
    # # build datasets
    if data_source == 'online': # from online dataset
        resize = (28, 28)
        fmnist_data_dir = configs['fmnist_data_dir']
        trainset = FMNISTDatasetOnline(path=fmnist_data_dir, is_train=True, resize=resize)
        validset = FMNISTDatasetOnline(path=fmnist_data_dir, is_train=False, resize=resize)
        testset = FMNISTDatasetOnline(path=fmnist_data_dir, is_train=False, resize=resize)
    elif data_source == 'local': # from local data
        trainset = FMNISTDatasetLocal(configs['train_data'], configs['train_label'])
        validset = FMNISTDatasetLocal(configs['valid_data'], configs['valid_label'])
        testset = FMNISTDatasetLocal(configs['test_data'], configs['test_label'])
    else:
        raise ValueError(f'wrong data_source param {data_source}. must be one of online/local')

    # cache image args
    img_shape = trainset.img_shape
    with open(image_args, 'w')  as f:
        json.dump(img_shape, f)

    # design net & loss
    vit = ViTEncoder(img_shape, patch_size, num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens)

    loss = nn.CrossEntropyLoss(reduction='none')

    # init trainer
    trainer = vitTrainer(vit, loss, num_epochs, batch_size)

    trainer.set_device(torch.device('cpu'))## set the device
    trainer.set_data_iter(trainset, validset, testset)## set the data iters
    trainer.set_optimizer(lr)## set the optimizer
    # no grad clipper
    trainer.set_epoch_eval(vitEpochEvaluator(num_epochs, train_logs_fpath, verbose=True))## set the epoch evaluator

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








def infer_job(saved_params_fpath):
    print('infer job begin')
        
    # load cached image args
    with open(image_args, 'r') as f:
        img_shape = json.load(f)

    # set device
    device = torch.device('cuda')

    ## construct model
    net = ViTEncoder(img_shape, patch_size, num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens).to(device)
    net.load_state_dict(torch.load(saved_params_fpath, map_location=device))
    net.eval()

    # set predictor
    classifier = fmnistClassifier(net, device)

    # predict
    classifier.predict(configs['test_data'], select_size=16, view=True)

    # evaluate output
    print( 'accuracy rate: ', classifier.evaluate(configs['test_label'], view=True) )

    print('pred scores: ', classifier.pred_scores)

    print('infer job complete')
    return