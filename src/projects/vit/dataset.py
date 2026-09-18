import torch
import torchvision
from torchvision import transforms
from src.utils.image.mnist import decode_idx3_ubyte, decode_idx1_ubyte



class FMNISTDatasetOnline(torch.utils.data.Dataset):
    def __init__(self, path, is_train, resize):
        super().__init__()
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self._fmnist = torchvision.datasets.FashionMNIST(root=path, train=is_train, transform=trans, download=False)
        self._img_shape = self._fmnist[0][0].shape[1:]

    def __getitem__(self, index):
        return self._fmnist[index]
    
    def __len__(self):
        return len(self._fmnist)
    
    @property
    def img_shape(self):
        return self._img_shape


class FMNISTDatasetLocal(torch.utils.data.Dataset):
    def __init__(self, imgdata_fpath, imglabel_fpath):
        super().__init__()
        self._imgTensor = decode_idx3_ubyte(imgdata_fpath).type(torch.float32) #shape: (numImgs, 1, numRows, numCols)
        self._labelTensor = decode_idx1_ubyte(imglabel_fpath).type(torch.int64) #shape: (numImgs, )
        self._img_shape = self._imgTensor.shape[1:]
        
        assert self._imgTensor.size(0) == self._labelTensor.size(0), 'image data & label sample size mismatch'
    
    def __getitem__(self, index):
        return (self._imgTensor[index], self._labelTensor[index])

    def __len__(self):
        return self._imgTensor.size(0)

    @property
    def img_shape(self):
        return self._img_shape