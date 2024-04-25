import torch.nn as nn
import math
import torch
import numpy as np

"""
每个NN都必须继承nn.Module(可以继承再继承)
每个NN都必须至少实现__init__和forward两个方法. 其中__init__确定了拓扑结构, forward确定了前向过程(同时也确定了反向传播过程)
torch.nn已经定义了一些常用的基础NN, 同时torch / torch.nn.functional定义了一些梯度可传的函数操作.
所以我们搭建NN的过程,就是用torch.nn完成基础层, 然后用torch.nn.functional完成一些更精细的操作.
最后一些常用操作也可以帮忙控制前向过程/梯度反传过程, 比如indexput操作, 可以让被put的位置不参与相关参数的更新
"""
class TestNN(nn.Module):
    """
    这是一个测试NN, 用来快速复习搭建NN的各个函数
    """
    def __init__(self, onehot_size, hidden_size, out_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(onehot_size, hidden_size)
        # self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, out_size)

    ## 经典的带参数前向计算 layers

    # nn.Embedding
    # nn.Embedding module 相当于 将 n 维tensor的值, 作onehot处理后, 成为 n+1 维的tensor
    # 此时最后一维 的dim size 是 onehot_size(高维稀疏空间). 随后乘以一个 onehot_size 到 hidden_size 的线性映射
    # 最终输出 n+1 维 的tensor. 相当于原 tensor 的每个scalar值 被映射到 hidden_size 维的低维空间

    # nn.Linear
    # nn.Linear 有input_size 和 output_size, 要求输入的data tensor 的最后一维是input_size, 输出data tensor的最后一维是 output_size

    # nn.LazyLinear
    # nn.LazyLinear 只有output_size这一个参数. 对输入的data tensor的维度没有要求，它会自动根据输入的last dim来作 input_size

    ## 泛化随机 dropout layers
    # nn.Dropout
    # nn.Dropout 取 dropout_rate (p) 作为参数, 在训练的时候以p的概率 de-activate 某个神经元(即set to 0), 然后将该神经元的值 dilate by 1/(1-p)
    # 提高泛化性

    ## 非线形 layers
    # nn.ReLU / nn.Sigmoid / nn.Tanh / nn.GeLU 等等. 一般来说，为了纯粹的非线形激活, 还是多采用 nn.functional 函数的办法激活

    ## 规范化 Normalization Layers
    # nn.LayerNorm / nn.BatchNorm 等等

    ## 其他用在 图像 等数据上的 Layers
    # Convolution / Pooling / Padding 等

    def forward(self, X):
        X_embd = self.embedding(X)
        Y = self.linear(nn.functional.relu(X_embd))
        return Y
    






if __name__ == "__main__":
    X = torch.randint(low=0, high=9, size=(3,4,5), dtype=torch.int)
    tnn = TestNN(onehot_size=10, hidden_size=3, out_size=5)
    Y = tnn(X)
    print(Y)
