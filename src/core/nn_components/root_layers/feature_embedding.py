import torch.nn as nn
from torch import Tensor
import torch



def offset_multifeatures(input_tensor:Tensor, num_classes:Tensor):
    '''
    input_tensor: shape as [..., num_categorical_features]
        categorical features 排列在 last dim, 共有 num_categorical_features 个 feature
    
    num_classes: 1D tensor (num_categorical_features, ), dtype=torch.int64
        依 input_tensor last dim的顺序, 指明每个feature的category size
    '''
    assert num_classes.shape[0] == input_tensor.shape[-1], 'every feature must have its num_class'

    assert torch.all(input_tensor < num_classes), \
        f'Categorical Features(as index number) must be strictly smaller than corresponding num_class'
    
    # offsets: 1-D tensor (num_categorical_features,) as [0, num_categs_for_ft1+num_categs_for_ft2, ...]
    # offsets shift feature 1 with 0, shift feature 2 with num_categs_of_feature1, shift feature 3 with num_categs_of_feature1and2,...
    offsets = torch.cat(
        [torch.zeros(1, device=num_classes.device), torch.cumsum(num_classes, dim=0)[:-1]], dim=0
        ).to(dtype=input_tensor.dtype, device=input_tensor.device)
    
    # broadcast at input_tensor's last dimenstion
    return input_tensor + offsets



class MultiCategFeatEmbedding(nn.Module):
    '''
    Embedding layer for multiple categorical features which had already been index-preprocessed.
    Compare with vanilla nn.Embedding:
        Embedding: embeds different values of the same categorical feature into different vectors
        MultiCategFeatEmbedding:
            embeds different values of different categorical features(1 value per each feature) into different independent vectors
    
    If one wants to use Emebedding for different values of different categorical features(1 value per each feature),
    one can offset right shift values by sum of number of value_sizes of previous categorical features.
    
    See `offset_multifeatures` function or `onehot_concat_multifeatures` function in Utils/Data/DataTransform

    args:
        num_classes (Tensor): shape (num_features, ), with elements are number of levels(classes) for every categorical feature
        embed_dim (int): the size of each feature's embedding vector
        flatten (Optional, Bool): If True, embedding output will flatten all features' embedded tensors.
                                  Ouput last dim will be "num_features * embed_dim"
    
    input:
        input (Tensor): shape (*, num_features), with elements are level-index of categorical features

    output:
        if flatten is True:
            returned tensor with shape (*, num_features * embed_dim)
        else:
            returned tensor with shape (*, num_features, embed_dim)
    '''
    def __init__(self, num_classes: Tensor, embed_dim: int, flatten: bool = True, *args, **kwargs):
        super().__init__()
        self.register_buffer('num_classes', num_classes, persistent=False)
        self.flatten = flatten
        self.embedding = nn.Embedding(num_classes.sum().item(), embed_dim, *args, **kwargs)
    
    def forward(self, input: Tensor):
        # input shape: (*, num_features)
        input_ = offset_multifeatures(input, self.num_classes)
        embed_ = self.embedding(input_)
        if self.flatten:
            return embed_.flatten(start_dim=-2) # shape: (*, num_features*embed_dim)
        else:
            return embed_ # shape: (*, num_features, embed_dim)