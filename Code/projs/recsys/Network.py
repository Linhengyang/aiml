import torch.nn as nn
import torch
import math
from ...Modules._recsys import MaskedMatrixFactorization, UnMaskMatrixFactorization
from ...Base.RootLayers.MultiCategFeatEmbedding import MultiCategFeatEmbedding

class MatrixFactorization(nn.Module):
    def __init__(self, num_factors, num_users, num_items):
        super().__init__()
        self.mf = UnMaskMatrixFactorization(num_factors, num_users, num_items)
    
    def forward(self, users_idx, items_idx):
        # shapes: (batch_size,)int64, (batch_size,)int64
        return self.mf(users_idx, items_idx)
    
    def __str__(self):
        return self.mf.__str__()

class ItemBasedAutoRec(nn.Module):
    def __init__(self, num_hiddens, num_users, dropout=0.05):
        super().__init__()
        self.encoder = nn.Linear(num_users, num_hiddens)
        self.act = nn.Sigmoid()
        self.decoder = nn.Linear(num_hiddens, num_users)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_batch_matrix):
        # input shape: (batch_size=num_items_batch, num_users)
        pred = self.decoder(self.dropout(self.act(self.encoder(input_batch_matrix))))
        if self.training:
            return pred * torch.sign(input_batch_matrix)
        else:
            return pred

class FactorizationMachine(nn.Module):
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # input = offset_multifeatures(input, self.num_classes)
        # input shape: (batch_size, num_features). each feature has its own value_size stored in num_classes
        bias = self.global_bias.expand(input.shape[0]) # (1,) -> (batch_size, )
        linear = self.linear_embedding(input).squeeze(2).sum(dim=1) #  shape (batch_size, num_features, 1) -> (batch_size,)
        embedd_ = self.quadratic_embedding(input) # shape (batch_size, num_features, num_factor)
        square_of_sum = embedd_.sum(dim=1).pow(2) # shape (batch_size, num_factor)
        sum_of_square = (embedd_.pow(2)).sum(dim=1) # shape (batch_size, num_factor)
        quadratic = 0.5 * (square_of_sum - sum_of_square).sum(dim=1) # (batch_size, num_factor) -> (batch_size,)
        return self.sigmoid(bias + linear + quadratic)
