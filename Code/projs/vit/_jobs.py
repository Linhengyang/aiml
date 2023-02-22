import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from .Dataset import FMNISTDatasetOnline, FMNISTDatasetLocal
from .Network import ViTEncoder
from .Trainer import vitTrainer
from .Evaluator import vitEpochEvaluator
from .Predictor import fmnistClassifier
import yaml
configs = yaml.load(open('Code/projs/vit/configs.yaml', 'rb'), Loader=yaml.FullLoader)
local_model_save_dir = configs['local_model_save_dir']
base_data_dir = configs['base_data_dir']
fmnist_dir = configs['fmnist_dir']
fmnist_train_img_fname = configs['fmnist_train_img_fname']
fmnist_train_label_fname = configs['fmnist_train_label_fname']
fmnist_valid_img_fname = configs['fmnist_valid_img_fname']
fmnist_valid_label_fname = configs['fmnist_valid_label_fname']
fmnist_test_img_fname = configs['fmnist_test_img_fname']
fmnist_test_label_fname = configs['fmnist_test_label_fname']

def train_job():
    # # build datasets from online dataset
    # path = '../../data'
    # resize = (96, 96)
    # trainset = FMNISTDatasetOnline(path, True, resize)
    # validset = FMNISTDatasetOnline(path, False, resize)
    # testset = FMNISTDatasetOnline(path, False, resize)

    # build dataset from local data
    train_img_path = os.path.join([base_data_dir, fmnist_dir, fmnist_train_img_fname])
    train_label_path = os.path.join([base_data_dir, fmnist_dir, fmnist_train_label_fname])
    valid_img_path = os.path.join([base_data_dir, fmnist_dir, fmnist_valid_img_fname])
    valid_label_path = os.path.join([base_data_dir, fmnist_dir, fmnist_valid_label_fname])
    test_img_path = os.path.join([base_data_dir, fmnist_dir, fmnist_test_img_fname])
    test_label_path = os.path.join([base_data_dir, fmnist_dir, fmnist_test_label_fname])
    trainset = FMNISTDatasetLocal(train_img_path, train_label_path)
    validset = FMNISTDatasetLocal(valid_img_path, valid_label_path)
    testset = FMNISTDatasetLocal(test_img_path, test_label_path)

    # design net & loss
    num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens = 2, 4, 64, 0.1, 0.1, 128
    img_shape, patch_size = trainset.img_shape, (7, 7)
    vit = ViTEncoder(img_shape, patch_size, num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens)
    loss = nn.CrossEntropyLoss(reduction='none')
    # init trainer for num_epochs & batch_size & learning rate
    num_epochs, batch_size, lr = 10, 128, 0.0005
    trainer = vitTrainer(vit, loss, num_epochs, batch_size)
    trainer.set_device(torch.device('cuda'))## set the device
    trainer.set_data_iter(trainset, validset, testset)## set the data iters
    trainer.set_optimizer(lr)## set the optimizer
    trainer.set_epoch_eval(vitEpochEvaluator(num_epochs, log_fname='train_logs.txt', visualizer=True))## set the epoch evaluator
    # start
    trainer.log_topology('lazy_topo.txt')## print the vit topology
    check_flag = trainer.resolve_net(need_resolve=True)## check the net & loss
    if check_flag:
        trainer.log_topology('def_topo.txt')## print the defined topology
        trainer.init_params()## init params
    # fit
    trainer.fit()
    # save
    trainer.save_model('vit_v1.params')

def infer_job():
    device = torch.device('cpu')
    # obtain pred batch
    path = '../../data'
    resize = (28, 28)
    validset = FMNISTDatasetOnline(path, False, resize)
    for img_batch, labels in validset:
        break
    # load model
    num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens = 2, 4, 512, 0.3, 0.4, 512
    img_shape, patch_size = (1, 28, 28), (7, 7)
    net = ViTEncoder(img_shape, patch_size, num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens)
    trained_net_path = os.path.join(local_model_save_dir, 'vit', 'vit_v1.params')
    net.load_state_dict(torch.load(trained_net_path, map_location=device))
    # init predictor
    classifier = fmnistClassifier(device)
    # predict
    print(classifier.predict(img_batch, labels, net))
    # evaluate
    print('accuracy: ', classifier.evaluate())
    print('pred scores: ', classifier.pred_scores)