import os
import warnings
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
from Code.Loss.L2PenaltyMSELoss import L2PenaltyMSELoss
from Code.projs.cfrec.Network import MatrixFactorization

if __name__ == "__main__":
    '''
    test code in this test.py file. After successful tests, code will be moved to _jobs.py under proj_name
    '''
    loss = L2PenaltyMSELoss(lambd=1000)
    num_factors = 2
    num_users = 3
    num_items = 4
    net = MatrixFactorization(num_factors, num_users, num_items)

    users_idx = torch.tensor([0, 0, 2, 2, 0])
    items_idx = torch.tensor([3, 0, 0, 3, 1])
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    interactions = torch.zeros(3, 4)
    interactions[users_idx, items_idx] = scores
    print('truth interactions: ', '\n', interactions)
    net_result = net(users_idx, items_idx)
    pred_interactions = net_result[0]
    weight_tensors = net_result[1:]
    l = loss(pred_interactions, interactions, *weight_tensors)
    l.backward()
    print('P grad for lambda 1000', net.user_factor_mat.grad)
    print('bi grad for lambda 1000', net.item_bias.grad)