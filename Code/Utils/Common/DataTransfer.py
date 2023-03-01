import torch
import numpy as np
import torch.nn as nn

def onehot_concat_multifeatures(input_tensor, num_classes):
    '''
    input:
    1. input_tensor:
    (*, num_features), with elements are level-index of categorical features
    The last dimention of input_tensor, is among different categorical features
    >>>
    tensor([[9, 3, 4],
            [0, 1, 0],
            [7, 0, 0],
            [0, 0, 0],
            [1, 1, 1]])
    where batch_size = 5, num_features = 3
    >>>
    tensor([9, 3, 4, 0])
    where num_features = 4
    >>>
    tensor([[[9, 3, 4],
             [0, 1, 0],
             [7, 0, 0],
             [0, 0, 0],
             [1, 1, 1]],
            [[9, 3, 4],
             [0, 1, 0],
             [7, 0, 0],
             [0, 0, 0],
             [1, 1, 1]]])
    where batch_size = 2, position_dims = 5, num_features = 3
    2. num_classes:
    (num_features, ), with elements are number of levels(classes) for every categorical feature
    
    len(num_classes) == input_tensor.shape[-1]

    return:
    onehot every catogorical feature along its own num_class, then concat all onehot vectors along on dim=-1
    shape: ( *, sum(num_classes) )
    '''
    assert len(num_classes) == input_tensor.shape[-1], 'every feature must have its num_class'
    assert torch.all(input_tensor < num_classes), 'index number exceeds or be equal to num_classes. Index number must be smaller than corresponding num_class'
    offsets = torch.cat([torch.zeros(1,), torch.cumsum(num_classes, dim=0)[:-1]], dim=0).type(num_classes.dtype).to(input_tensor.device)
    return nn.functional.one_hot(input_tensor + offsets, num_classes.sum()).sum(dim=-2)

def offset_multifeatures(input_tensor, num_classes):
    assert len(num_classes) == input_tensor.shape[-1], 'every feature must have its num_class'
    assert torch.all(input_tensor < num_classes), 'index number exceeds or be equal to num_classes. Index number must be smaller than corresponding num_class'
    offsets = torch.cat([torch.zeros(1,), torch.cumsum(num_classes, dim=0)[:-1]], dim=0).type(num_classes.dtype).to(input_tensor.device)
    return (input_tensor + offsets).type(input_tensor.dtype)
