import torch.nn as nn
from torch import Tensor
import torch
import warnings

def offset_multifeatures(input_tensor, num_classes):
    assert len(num_classes) == input_tensor.shape[-1], 'every feature must have its num_class'
    assert torch.all(input_tensor < num_classes), 'index number exceeds or be equal to num_classes. Index number must be smaller than corresponding num_class'
    offsets = torch.cat([torch.zeros(1,), torch.cumsum(num_classes, dim=0)[:-1]], dim=0).type(num_classes.dtype)
    return (input_tensor + offsets).type(input_tensor.dtype)

class MultiCategFeatEmbedding(nn.Embedding):
    '''
    Embedding layer for multiple categorical features which had already been index-preprocessed.
    Compare with vanilla nn.Embedding:
        Embedding: embeds different values of the same categorical feature into different vectors
        MultiCategFeatEmbedding: embeds different values of different categorical features(1 value per each feature) into different independent vectors
    
    If one wants to use Emebedding for different values of different categorical features(1 value per each feature), one can offset right shift values by 
    sum of number of value_sizes of previous categorical features.
    See `offset_multifeatures` function or `onehot_concat_multifeatures` function in Utils/DataTransform

    args:
        num_embeddings (int): shall be sum{num_classes of all features}. If not, a warning will be printed and num_embeddings will be reset automatically
        embedding_dim (int): the size of each feature's embedding vector
    
    input:
        input (Tensor): shape (*, num_features), with elements are level-index of categorical features
        num_classes (Tensor): shape (num_features, ), with elements are number of levels(classes) for every categorical feature
        flatten (Optional, Bool): If True, embedding output will flatten all features' embedded tensors. Ouput last dim will be "num_features * embedding_dim"
    
    output:
        if flatten is True:
            returned tensor with shape (*, num_features * embedding_dim)
        else:
            returned tensor with shape (*, num_features, embedding_dim)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, input: Tensor, num_classes: Tensor, flatten=True) -> Tensor:
        input_ = offset_multifeatures(input, num_classes)
        num_embeddings = int(num_classes.sum())
        if self.num_embeddings != num_embeddings:
            warnings.warn(f'arg num_embeddings must be the sum of number of classes of all features. num_embeddings={num_embeddings} set automatically')
            self.num_embeddings = num_embeddings
        embed_ = super(MultiCategFeatEmbedding, self).forward(input_)
        if flatten:
            return embed_.flatten(start_dim=-2)
        else:
            return embed_