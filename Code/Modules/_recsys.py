import torch.nn as nn
import torch
import math
from ..Base.RootLayers.MultiCategFeatEmbedding import MultiCategFeatEmbedding

def interaction_weights(U, I, users_idx, items_idx):
    weights = torch.zeros(U, I, device=users_idx.device)
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

class MaskedMatrixFactorization(nn.Module):
    '''
    Matrix Factorization with Mask, only observed users & items's latent factors will be updated through every batch

    agrs:
        num_factos, num_users, num_items

    inputs:
        user_idx: (batch_size,)int64 tensor
        item_idx: (batch_size,)int64 tensor
    
    outputs:
        S_hat: masked hat ratings, (num_users, num_items)
        P: masked user-factor weight, (num_users, num_factors)
        bu: masked user-bias bias, (num_users, 1)
        Q: masked item-factor weight, (num_items, num_factors)
        bi: masked item-bias bias, (num_items, )
    '''
    def __init__(self, num_factors, num_users, num_items):
        super().__init__()
        self.U = num_users
        self.I = num_items
        self.K = num_factors
        self.user_factor_weight = nn.Parameter(torch.randn(num_users, num_factors))
        self.item_factor_weight = nn.Parameter(torch.randn(num_items, num_factors))
        self.user_bias = nn.Parameter(torch.zeros(num_users, 1))
        self.item_bias = nn.Parameter(torch.zeros(num_items, ))
    def forward(self, users_idx, items_idx):
        # shapes: (batch_size,)int64, (batch_size,)int64
        interaction_w = interaction_weights(self.U, self.I, users_idx, items_idx) # (U, I)
        user_factor_w = factor_bias_weights(self.user_factor_weight, users_idx)
        item_factor_w = factor_bias_weights(self.item_factor_weight, items_idx)
        user_bias_w = factor_bias_weights(self.user_bias, users_idx)
        item_bias_w = factor_bias_weights(self.item_bias, items_idx)
        # 用乘以weights矩阵的方式, 保证只有这个批次中的user_idx和item_idx的相关参数被更新
        # 也可以尝试用indexput的方式
        # mask = torch.ones(U, I)
        # mask[users_idx, items_idx] = 0
        # mask = maks.type(torch.bool)
        # return S_hat[mask] = 0
        S_hat = torch.mm(self.user_factor_weight, self.item_factor_weight.transpose(1,0)) + self.user_bias + self.item_bias
        return (S_hat * interaction_w,
                self.user_factor_weight * user_factor_w, self.user_bias * user_bias_w,
                self.item_factor_weight * item_factor_w, self.item_bias * item_bias_w)
    def __str__(self):
        return "user_factor_matrix: {}; item_factor_matrix: {}".format(self.user_factor_weight.shape, self.item_factor_weight.shape)

class UnMaskMatrixFactorization(nn.Module):
    '''
    Matrix Factorization without Mask, all users & items's latent factors will be updated through every observed batch

    agrs:
        num_factos, num_users, num_items

    inputs:
        user_idx: (batch_size,)int64 tensor
        item_idx: (batch_size,)int64 tensor
    
    outputs:
        scores_hat: (batch_size, )float32 tensor
    '''
    def __init__(self, num_factors, num_users, num_items):
        super().__init__()
        self.user_factor_weight = nn.Embedding(num_users, num_factors)
        self.item_factor_weight = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, users_idx, items_idx):
        # shapes: (batch_size,)int64, (batch_size,)int64
        P_u = self.user_factor_weight(users_idx)
        Q_i = self.item_factor_weight(items_idx)
        b_u = self.user_bias(users_idx)
        b_i = self.item_bias(items_idx)
        # return ( (batch_size,), )
        return ( ((P_u * Q_i).sum(dim=1, keepdim=True) + b_u + b_i).flatten(), )
    
class QuadraticFactorizationMachine(nn.Module):
    def __init__(self, num_classes, num_factor):
        '''
        num_classes: tensor (value_size_of_feat1, value_size_of_feat2, ..., value_size_of_featN)
        sample: tensor (feat1_value, feat2_value, ..., featN_value)
        '''
        super().__init__()
        self.num_classes = num_classes
        self.quadratic_embedding = MultiCategFeatEmbedding(num_classes, num_factor, False)
        self.linear_embedding = MultiCategFeatEmbedding(num_classes, 1, False)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        # input = offset_multifeatures(input, self.num_classes)
        # input shape: (batch_size, num_features). each feature has its own value_size stored in num_classes
        bias = self.global_bias.expand(input.shape[0]) # (1,) -> (batch_size, )
        linear = self.linear_embedding(input).squeeze(2).sum(dim=1) #  shape (batch_size, num_features, 1) -> (batch_size,)
        embedd_ = self.quadratic_embedding(input) # shape (batch_size, num_features, num_factor)
        square_of_sum = embedd_.sum(dim=1).pow(2) # shape (batch_size, num_factor)
        sum_of_square = (embedd_.pow(2)).sum(dim=1) # shape (batch_size, num_factor)
        quadratic = 0.5 * (square_of_sum - sum_of_square).sum(dim=1) # (batch_size, num_factor) -> (batch_size,)
        return quadratic, linear, bias