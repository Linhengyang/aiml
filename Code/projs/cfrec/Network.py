import torch.nn as nn
import torch
import math

def interaction_weights(U, I, users_idx, items_idx):
    weights = torch.zeros(U, I)
    weights[users_idx, items_idx] = 1
    return weights

def factor_bias_weights(tensor, index):
    '''
    tensor可以是:
        1. user_factor_mat: (num_users, K)
        2. item_factor_mat: (num_items, K)
        3. user_bias: (num_users, 1)
        4. item_bias: (num_items, )
    '''
    weights = torch.zeros_like(tensor)
    weights[index] = 1
    return weights

class MatrixFactorization(nn.Module):
    def __init__(self, num_factors, num_users, num_items):
        super().__init__()
        self.U = num_users
        self.I = num_items
        self.K = num_factors
        self.user_factor_mat = nn.Parameter(torch.randn(num_users, num_factors))
        self.item_factor_mat = nn.Parameter(torch.randn(num_items, num_factors))
        self.user_bias = nn.Parameter(torch.zeros(num_users, 1))
        self.item_bias = nn.Parameter(torch.zeros(num_items, ))
    def forward(self, users_idx, items_idx):
        # shapes: (batch_size,)int64, (batch_size,)int64
        interaction_w = interaction_weights(self.U, self.I, users_idx, items_idx) # (U, I)
        user_factor_w = factor_bias_weights(self.user_factor_mat, users_idx)
        item_factor_w = factor_bias_weights(self.item_factor_mat, items_idx)
        user_bias_w = factor_bias_weights(self.user_bias, users_idx)
        item_bias_w = factor_bias_weights(self.item_bias, items_idx)
        # 用乘以weights矩阵的方式, 保证只有这个批次中的user_idx和item_idx的相关参数被更新
        # 也可以尝试用indexput的方式
        # mask = torch.ones(U, I)
        # mask[users_idx, items_idx] = 0
        # mask = maks.type(torch.bool)
        # return S_hat[mask] = 0
        S_hat = torch.mm(self.user_factor_mat, self.item_factor_mat.transpose(1,0)) + self.user_bias + self.item_bias
        return (S_hat * interaction_w,
                self.user_factor_mat * user_factor_w, self.user_bias * user_bias_w,
                self.item_factor_mat * item_factor_w, self.item_bias * item_bias_w)
        