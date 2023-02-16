from torch.utils import data

def tensors2batch_iter(data_arrays, batch_size, is_train=True):
    """
    inputs: data_arrays, batch_size, is_train(optional)
        data_arrays: container, consisted of multiple parts with same length, e.g (features, labels, indices,...)
        batch_size: batch_size for every batch
        is_train: whether to shuffle sample. True for train mode

    returns: denoted as data_iter
        data_iter: A iterator, who gives minibatch of data with shape (batch_size, *, *, ..) at each iter
    
    explains:
        组装features和labels等, 给出minibatch data loader
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)