import os
import warnings
warnings.filterwarnings("ignore")
import torch
import math
import torch.nn as nn
from .Dataset import MovieLensRatingDataset, d2lCTRDataset
from .Network import MatrixFactorization, ItemBasedAutoRec, FactorizationMachine
from ...Loss.L2PenaltyMSELoss import L2PenaltyMSELoss
from .Trainer import mfTrainer, autorecTrainer, fmTrainer
from .Evaluator import mfEpochEvaluator, autorecEpochEvaluator, fmEpochEvaluator
from .Predictor import MovieRatingMFPredictor
import yaml
configs = yaml.load(open('Code/projs/recsys/configs.yaml', 'rb'), Loader=yaml.FullLoader)
local_model_save_dir = configs['local_model_save_dir']
base_data_dir = configs['base_data_dir']
movielens_dir = configs['movielens_dir']
movielens_fname = configs['movielens_fname']
d2lctr_dir = configs["d2lctr_dir"]
d2lctr_train_fname = configs['d2lctr_train_fname']
d2lctr_valid_fname = configs['d2lctr_valid_fname']
d2lctr_test_fname = configs['d2lctr_test_fname']

def fm_train_job():
    # build dataset from local data
    trainset = d2lCTRDataset( os.path.join(base_data_dir, d2lctr_dir, d2lctr_train_fname) )
    validset = d2lCTRDataset( os.path.join(base_data_dir, d2lctr_dir, d2lctr_valid_fname) )
    testset = d2lCTRDataset( os.path.join(base_data_dir, d2lctr_dir, d2lctr_test_fname) )
    # design net & loss
    num_factor = 20
    net = FactorizationMachine(trainset.num_classes, num_factor)
    loss = nn.BCELoss()
    # init trainer for num_epochs & batch_size & learning rate
    num_epochs, batch_size, lr = 100, 1024, 0.0005
    trainer = fmTrainer(net, loss, num_epochs, batch_size)
    trainer.set_device(torch.device('cuda'))## set the device
    trainer.set_data_iter(trainset, validset, testset)## set the data iters
    trainer.set_optimizer(lr)## set the optimizer
    trainer.set_epoch_eval(fmEpochEvaluator(num_epochs, 'fm_train_logs.txt', visualizer=True))## set the epoch evaluator
    # start
    check_flag = trainer.resolve_net(need_resolve=True)## check the net & loss
    if check_flag:
        trainer.log_topology('factorization_machine.txt')## print the defined topology
        trainer.init_params()## init params
    # fit
    trainer.fit()
    # save
    trainer.save_model('factorization_machine_k5_v1.params')

def mf_train_job():
    # build dataset from local data
    data_path = os.path.join(base_data_dir, movielens_dir, movielens_fname)
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
    trainer.save_model('matrix_factorization_k5_v1.params')

def mf_infer_job():
    device = torch.device('cpu')
    data_path = os.path.join(base_data_dir, movielens_dir, movielens_fname)
    validset = MovieLensRatingDataset(data_path, False, 'random', seed=0)
    valid_iter = torch.utils.data.DataLoader(validset, 10, True)
    for users, items, scores in valid_iter:
        break
    # load model
    num_users = validset.num_users
    num_items = validset.num_items
    num_factors = 5
    net = MatrixFactorization(num_factors, num_users, num_items)
    trained_net_path = os.path.join(local_model_save_dir, 'recsys', 'matrix_factorization_k5_v1.params')
    net.load_state_dict(torch.load(trained_net_path, map_location=device))
    # init predictor
    rater = MovieRatingMFPredictor(device)
    # predict
    print('truth ratings: ', scores)
    print('pred ratings: ', rater.predict(users, items, net))
    # evaluate
    print('rmse: ', rater.evaluate(scores))

def autorec_train_job():
    # build dataset from local data
    data_path = os.path.join(base_data_dir, movielens_dir, movielens_fname)
    train_dataset = MovieLensRatingDataset(data_path, True, 'random')
    trainset = train_dataset.interactions_itembased
    valid_dataset = MovieLensRatingDataset(data_path, False, 'random')
    validset = valid_dataset.interactions_itembased
    test_dataset = MovieLensRatingDataset(data_path, False, 'random')
    testset = test_dataset.interactions_itembased
    # design net & loss
    num_users = train_dataset.num_users
    num_items = train_dataset.num_items
    num_factors = 5
    net = ItemBasedAutoRec(num_factors, num_users)
    loss = L2PenaltyMSELoss(0.1)
    # init trainer for num_epochs & batch_size & learning rate
    num_epochs, batch_size, lr = 200, 128, 0.00005
    trainer = autorecTrainer(net, loss, num_epochs, batch_size)
    trainer.set_device(torch.device('cuda'))## set the device
    trainer.set_data_iter(trainset, validset, testset)## set the data iters
    trainer.set_optimizer(lr)## set the optimizer
    trainer.set_epoch_eval(autorecEpochEvaluator(num_epochs, 'itembased_autorec_train_logs.txt', visualizer=False))## set the epoch evaluator
    # start
    check_flag = trainer.resolve_net(need_resolve=True)## check the net & loss
    if check_flag:
        trainer.log_topology('itembased_autorec.txt')## print the defined topology
        trainer.init_params()## init params
    # fit
    trainer.fit()
    # save
    trainer.save_model('itembased_autorec_k5_v1.params')

def autorec_infer_job():
    device = torch.device('cpu')
    data_path = os.path.join(base_data_dir, movielens_dir, movielens_fname)
    test_dataset = MovieLensRatingDataset(data_path, False, 'random')
    testset = test_dataset.interactions_itembased
    # design net & loss
    num_users = test_dataset.num_users
    num_items = test_dataset.num_items
    num_factors = 5
    net = ItemBasedAutoRec(num_factors, num_users)
    trained_net_path = os.path.join(local_model_save_dir, 'recsys', 'itembased_autorec_k5_v1.params')
    net.load_state_dict(torch.load(trained_net_path, map_location=device))
    net.eval()
    with torch.no_grad():
        pred_score_matrix = net(testset)
    for users, items, scores in torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False):
        break
    pred_scores = pred_score_matrix[users, items]
    print('truth scores: ', scores)
    print('pred scores: ', pred_scores)
    print('rmse: ', math.sqrt(torch.sum( (pred_scores - scores).pow(2) )))