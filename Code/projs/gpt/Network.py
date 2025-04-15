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
        self.embedding_1 = nn.Embedding(onehot_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, out_size)

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

    ## 以上构成了torch模型的一些经典layers
    ## 在内部, 可以使用私有属性 _modules 访问
        # print("_modules: ", self._modules)
        # print("**************************************")
    
    ## 但是很多时候, 模型需要微操至tensor. torch提供多种方式添加tensor. 如果该tensor是需要被更新的, 那么它是parameter, 如果不需要, 那么它是buffer
    
    ## 添加 不需要被更新的 buffer
        # 方式0: 成员变量
        # self.buffer = torch.tensor(1.) 这种方式是不行的, 该值不会被注册进入模型, 无法跟随 state_dict 表述, 也不能跟随整个模型在device之间移动
        # 唯一的方式
        self.register_buffer("my_buffer", torch.tensor(1.) )
        # 可以使用私有属性 _buffers 访问
        # print("_buffers: ", self._buffers)
        # 在外部, 可以使用方法 named_buffers()/ buffers 来展示

    ## 添加 需要被更新的 parameter
        # 有两种写法，效果几乎完全相同
        # 都是创建新 parameter 变量, 并注册到模型中; 都需要使用 nn.Parameter()方法来确定梯度
        self.register_parameter("my_param1", nn.Parameter(torch.tensor(1.)) )
        self.my_param2 = nn.Parameter(torch.tensor(2.))
        # 在内部, 可以使用私有属性 _parameters 访问
        # print("_parameters: ", self._parameters)
        # print("**************************************")
        # 在外部, 可以使用方法 named_parameters()/ parameters() 来展示.
        # _parameters 只会返回 额外注册的 parameters, 通过 _modules 注册的参数不会被访问
        # 而 外部的 named_parameters()/ parameters() 会返回 _parameters 和 _modules 所有可学习参数



    def forward(self, X):
        X_embd = self.embedding_1(X)
        Y = self.linear_2(nn.functional.relu(X_embd))
        return Y
    






if __name__ == "__main__":
    X = torch.randint(low=0, high=9, size=(3,4,5), dtype=torch.int)
    tnn = TestNN(onehot_size=10, hidden_size=3, out_size=5)
    Y = tnn(X)
    print(  list( tnn.named_parameters() ) )
