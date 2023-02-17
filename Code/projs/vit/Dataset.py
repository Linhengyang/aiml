from ...Utils.Text.TextPreprocess import preprocess_space
from ...Utils.Text.TextPreprocess import Vocab
from ...Utils.Common.SeqOperation import truncate_pad
import torch
class FashionMNISTDataset(torch.utils.data.Dataset):
    def __init__(self):
        raise NotImplementedError
    
    def __getitem__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
