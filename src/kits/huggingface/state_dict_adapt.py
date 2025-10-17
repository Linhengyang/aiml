# state_dict_adapt 里修改 huggingface/transformers 库标准的 model state_dict(weights), 以适应并载入 custom network
from collections import OrderedDict
from torch import Tensor

def gpt2_state_dict_adaptor(state_dict: OrderedDict[str, Tensor]) -> OrderedDict[str, Tensor]:
    pass