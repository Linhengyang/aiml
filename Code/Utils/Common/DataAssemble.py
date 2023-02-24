from torch.utils import data

def tensors2batch_iter(data_tensors, batch_size, is_train=True):
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
    dataset = data.TensorDataset(*data_tensors)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)