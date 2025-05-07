import torch
import torchvision
from torchvision import transforms
import numpy as np
import struct
from torch import Tensor


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



def decode_idx3_ubyte(file) -> Tensor:
    '''
    解析idx3数据文件: 图像, 返回(num_examples, 1, 28, 28)float的tensor
    '''
    # 读取二进制数据
    with open(file, 'rb') as fp:
        bin_data = fp.read()
    # 解析文件中的头信息. 从文件头部依次读取四个32位, 分别为: magic_num, numImgs, numRows, numCols(没有numChannels)
    # 偏置
    offset = 0
    # 读取格式: 大端
    fmt_header = '>iiii'
    magic_num, numImgs, numRows, numCols = struct.unpack_from(fmt_header, bin_data, offset)
    print('reading file ', file, ' with magic ', magic_num, ' image number ', numImgs)
    # 解析图片数据
    # 偏置掉头文件信息
    offset = struct.calcsize(fmt_header)
    # 读取格式
    fmt_image = '>'+str(numImgs*numRows*numCols)+'B'
    data = torch.tensor(struct.unpack_from(fmt_image, bin_data, offset)).reshape(numImgs, 1, numRows, numCols)

    return data



def decode_idx1_ubyte(file) -> Tensor:
    """
    解析idx1数据文件: 标签, 返回(num_examples, )int64的tensor
    """
    # 读取二进制数据
    with open(file, 'rb') as fp:
        bin_data = fp.read()
    # 解析文件中的头信息. 从文件头部依次读取两个个32位，分别为: magic，numImgs
    # 偏置
    offset = 0
    # 读取格式: 大端
    fmt_header = '>ii'
    magic_num, numImgs = struct.unpack_from(fmt_header, bin_data, offset)
    print('reading file ', file, ' with magic ', magic_num, ' image number ', numImgs)
    # 解析图片数据
    # 偏置掉头文件信息
    offset = struct.calcsize(fmt_header)
    # 读取格式
    fmt_image = '>'+str(numImgs)+'B'
    data = torch.tensor(struct.unpack_from(fmt_image, bin_data, offset))
    
    return data



class FMNISTDatasetLocal(torch.utils.data.Dataset):
    def __init__(self, imgdata_fpath, imglabel_fpath):
        super().__init__()
        self._imgTensor = decode_idx3_ubyte(imgdata_fpath).type(torch.float32) #shape: (numImgs, 1, numRows, numCols)
        self._labelTensor = decode_idx1_ubyte(imglabel_fpath).type(torch.int64) #shape: (numImgs, )
        self._img_shape = self._imgTensor.shape[1:]
        
        assert self._imgTensor.size(0) == self._labelTensor.size(0), 'sample size mismatch'
    
    def __getitem__(self, index):
        return (self._imgTensor[index], self._labelTensor[index])

    def __len__(self):
        return self._imgTensor.size(0)

    @property
    def img_shape(self):
        return self._img_shape