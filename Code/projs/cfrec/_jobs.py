import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from .Dataset import MovieLensRatingDataset
from .Network import MatrixFactorization
from ...Loss.L2PenaltyMSELoss import L2PenaltyMSELoss
from .Trainer import mfTrainer
from .Evaluator import mfEpochEvaluator
from .Predictor import MovieRatingPredictor
import yaml
configs = yaml.load(open('Code/projs/cfrec/configs.yaml', 'rb'), Loader=yaml.FullLoader)
local_model_save_dir = configs['local_model_save_dir']
base_data_dir = configs['base_data_dir']
movielens_dir = configs['movielens_dir']
data_fname = configs['data_fname']

def mf_train_job():
    # build dataset from local data
    data_path = os.path.join(base_data_dir, movielens_dir, data_fname)
    trainset = MovieLensRatingDataset(data_path, True, 'random')
    validset = MovieLensRatingDataset(data_path, False, 'random')
    testset = MovieLensRatingDataset(data_path, False, 'random')
    # design net & loss
    num_users = trainset.num_users
    num_items = trainset.num_items
    num_factors = 5
    net = MatrixFactorization(num_factors, num_users, num_items)
    loss = L2PenaltyMSELoss(0.1)
    # init trainer for num_epochs & batch_size & learning rate
    num_epochs, batch_size, lr = 200, 128, 0.00005
    trainer = mfTrainer(net, loss, num_epochs, batch_size)
    trainer.set_device(torch.device('cuda'))## set the device
    trainer.set_data_iter(trainset, validset, testset)## set the data iters
    trainer.set_optimizer(lr)## set the optimizer
    trainer.set_epoch_eval(mfEpochEvaluator(num_epochs, 'mf_train_logs.txt', visualizer=True))## set the epoch evaluator
    # start
    check_flag = trainer.resolve_net(need_resolve=True)## check the net & loss
    if check_flag:
        trainer.log_topology('matrix_factorization.txt')## print the defined topology
        trainer.init_params()## init params
    # fit
    trainer.fit()
    # save
    trainer.save_model('matrix_factorization_k10_v1.params')

def mf_infer_job():
    device = torch.device('cpu')
    data_path = os.path.join(base_data_dir, movielens_dir, data_fname)
    validset = MovieLensRatingDataset(data_path, False, 'random', seed=0)
    valid_iter = torch.utils.data.DataLoader(validset, 10, True)
    for users, items, scores in valid_iter:
        break
    # load model
    num_users = validset.num_users
    num_items = validset.num_items
    num_factors = 5
    net = MatrixFactorization(num_factors, num_users, num_items)
    trained_net_path = os.path.join(local_model_save_dir, 'cfrec', 'matrix_factorization_k5_v1.params')
    net.load_state_dict(torch.load(trained_net_path, map_location=device))
    # init predictor
    rater = MovieRatingPredictor(device)
    # predict
    print('truth ratings: ', scores)
    print('pred ratings: ', rater.predict(users, items, net))
    # evaluate
    print('rmse: ', rater.evaluate(scores))
