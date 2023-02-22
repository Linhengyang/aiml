from ...Utils.Text.TextPreprocess import preprocess_space
from ...Utils.Text.TextPreprocess import Vocab
from ...Utils.Common.SeqOperation import truncate_pad
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

class FMNISTDatasetOnline(torch.utils.data.Dataset):
    def __init__(self, path, is_train, resize):
        super().__init__()
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self._fmnist = torchvision.datasets.FashionMNIST(root=path, train=is_train, transform=trans, download=False)
    
    def __getitem__(self, index):
        return self._fmnist[index]
    
    def __len__(self):
        return len(self._fmnist)
