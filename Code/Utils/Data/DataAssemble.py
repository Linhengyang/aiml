from torch.utils import data
from torch import Tensor

def tensors2batch_iter(data_tensors: Tensor, batch_size: int, is_train=True):
    """
    inputs: data_tensors, batch_size, is_train(optional)
        data_tensors: container, consisted of multiple tensors(torch.tensor) with same length, e.g (features, labels, indices,...). 
        batch_size: batch_size for every batch
        is_train: whether to shuffle sample. True for train mode

    returns: denoted as data_iter
        data_iter: A iterator, who gives minibatch of data with shape (batch_size, *, *, ..) at each iter
    
    explains:
        组装features和labels等, 给出minibatch data loader
    """
    dataset = data.TensorDataset(*data_tensors) # 将长度相同的tensors组装成tensorDataset, 取i操作[i]直接返回tuple of 各tensors取i操作[i]的结果
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # 返回minibatch的loader, 每个循环返回batch_size个tuple of 各tensors取i操作的结果




