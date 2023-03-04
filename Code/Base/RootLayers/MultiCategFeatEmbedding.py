import torch.nn as nn
from torch import Tensor
import torch
import warnings

def offset_multifeatures(input_tensor :Tensor, num_classes :Tensor):
    '''
    input_tensor: (*, num_categorical_features)
    num_classes: (num_categorical_features, )
    '''
    num_classes = num_classes.to(input_tensor.device)
    assert num_classes.shape[0] == input_tensor.shape[-1], 'every feature must have its num_class'
    assert torch.all(input_tensor < num_classes), 'index number exceeds or be equal to num_classes. Index number must be smaller than corresponding num_class'
    offsets = torch.cat([torch.zeros(1, device=input_tensor.device), torch.cumsum(num_classes, dim=0)[:-1]], dim=0).type(num_classes.dtype)
    return (input_tensor + offsets).type(input_tensor.dtype)

class MultiCategFeatEmbedding(nn.Module):
    '''
    Embedding layer for multiple categorical features which had already been index-preprocessed.
    Compare with vanilla nn.Embedding:
        Embedding: embeds different values of the same categorical feature into different vectors
        MultiCategFeatEmbedding: embeds different values of different categorical features(1 value per each feature) into different independent vectors
    
    If one wants to use Emebedding for different values of different categorical features(1 value per each feature), one can offset right shift values by 
    sum of number of value_sizes of previous categorical features.
    See `offset_multifeatures` function or `onehot_concat_multifeatures` function in Utils/DataTransform

    args:
        num_classes (Tensor): shape (num_features, ), with elements are number of levels(classes) for every categorical feature
        embedding_dim (int): the size of each feature's embedding vector
        flatten (Optional, Bool): If True, embedding output will flatten all features' embedded tensors. Ouput last dim will be "num_features * embedding_dim"
    
    input:
        input (Tensor): shape (*, num_features), with elements are level-index of categorical features

    output:
        if flatten is True:
            returned tensor with shape (*, num_features * embedding_dim)
        else:
            returned tensor with shape (*, num_features, embedding_dim)
    '''
    def __init__(self, num_classes: Tensor, num_factor: int, flatten: bool, *args, **kwargs):
        super().__init__()
        self.register_buffer('num_classes', num_classes)
        self.flatten = flatten
        self.embedding = nn.Embedding(int(num_classes.sum()), num_factor, *args, **kwargs)
    
    def forward(self, input: Tensor):
        # input shape: (*, num_features)
        input_ = offset_multifeatures(input, self.num_classes)
        embed_ = self.embedding(input_) # shape: (*, num_features, num_factor)
        if self.flatten:
            return embed_.flatten(start_dim=-2) # shape: (*, num_features*num_factor)
        else:
            return embed_