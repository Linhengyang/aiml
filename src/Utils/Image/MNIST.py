# MNIST.py: MNIST related functions/tools
import torch
import struct
import numpy as np

def decode_idx3_ubyte(file, dtype:str='tensor') -> torch.Tensor|np.ndarray:
    '''
    解析idx3数据文件: 图像, 返回(num_examples, 1, 28, 28)float的tensor/array
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

    # 输出 data
    if dtype == 'tensor':
        data = torch.tensor(struct.unpack_from(fmt_image, bin_data, offset)).reshape(numImgs, 1, numRows, numCols)
    elif dtype == 'array':
        data = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape(numImgs, 1, numRows, numCols)
    else:
        raise NotImplementedError(f'dtype must be one of tensor/array')
    
    return data



def decode_idx1_ubyte(file, dtype:str='tensor') -> torch.Tensor|np.ndarray:
    """
    解析idx1数据文件: 标签, 返回(num_examples, )int64的tensor/array
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

    # 输出 data
    if dtype == 'tensor':
        data = torch.tensor(struct.unpack_from(fmt_image, bin_data, offset))
    elif dtype == 'array':
        data = np.array(struct.unpack_from(fmt_image, bin_data, offset))
    else:
        raise NotImplementedError(f'dtype must be one of tensor/array')
    
    return data