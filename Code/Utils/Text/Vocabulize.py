
import typing as t
from .TextPreprocess import count_corpus




class Vocab:
    '''
    args: tokens, min_freq(optional), reserved_tokens(optional)
        tokens: 1D or 2D list of tokens(words, chars, etc)
        min_freq: threshold to map rare tokens to <unk> to save memory space
        reserved_tokens: user-reserved special tokens. <unk> is added in default
    
    returns:
        A mapper whose main function is to map tokens(words, chars) to indices, or vice versa.
    
    explains:
        定义一个vocab类, 它提供将token映射到integer ID的功能, 并可以自定义忽视token的阈值, 以及预先设定的tokens
    '''
    # 初始化, 需要提供tokens, token去除阈值（频次小于该值的视作<unk>）, 预存tokens
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None, unk_token='<unk>'):
        if tokens is None: # 处理未输入tokens的情况, 用空列表
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = [] # 处理未输入预存tokens的情况
        counter = count_corpus(tokens) # 计算tokens出现频次
        # 将(token, frequency)组成的pair, 按照frequency从多到少排序记录在一个list里
        # sorted()函数, 输入一个iterable对象, 按照key参数排序, 默认从小到大排序, reverse=True则为从大到小排序）
        # 返回一个list, 元素是(key, value)对
        self._token_freqs = sorted(counter.items(), key=lambda x:x[1], reverse=True)
        
        # 建立从token到数字的双射
        ## 初始化token列表, 以便用index取出token, 即从 数字映射到token
        self.idx_to_token = [unk_token] + reserved_tokens
        ## 初始化token-idx字典, 以便用取value的方式取idx, 即从token映射到数字
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        ## 遍历_token_freqs中的所有token, 按序添加到self.idx_to_token, 并添加token-idx的kv pair到token_to_idx
        for token, freq in self._token_freqs:
            if freq < min_freq: # 如果该token出现的频次小于阈值, 则视该token为<unk>, 即已经建立好双射了. 后续的也不需要再建立映射, 跳出循环
                break
            else:
                if token not in [unk_token] + reserved_tokens: # <unk>和保留token已经在idx_to_token和token_to_idx中建立好双射关系了
                    self.idx_to_token.append(token) # 添加该token到self.idx_to_token
                    self.token_to_idx[token] = len(self.idx_to_token) - 1 # 添加token-idx的kv pair到token_to_idx
    
    def __len__(self):
        '''提供len()函数接口, 返回该vocab中总共有多少个distinct token'''
        return len(self.idx_to_token)
    
    @property # 修饰器, 在内部方法这样添加修饰之后, 可以以调用属性的方式来调用方法. 即 .unk() = .unk
    def unk(self):
        '''未知元 unk_token(默认为<unk>) 的idx为0'''
        return 0
    
    # 对给定的token（或者是tokens的hierarchy组合体）, 返回idx（或者是indices对应的hierarchy组合体）
    def __getitem__(self, tokens):
        '''提供文字to数字映射, 对给定的token, 返回其对应的idx. 如果输入有hierarchy结构, 则以相同的结构返回'''
        if not isinstance(tokens, (list, tuple)): # 当tokens不是容器类型时（即tokens是string时）
            # 字典的get方法, 取出tokens对应的value, 返回. 如果取不到, 即是未知元, 返回self.unk, 即0
            return self.token_to_idx.get(tokens, self.unk)
        # 当tokens是容器类型时, 取出内部的元素, 递归式地调用本函数. 这样本函数的返回值和输入值token具有相同的hierarchy结构
        return [self.__getitem__(token) for token in tokens]
    
    # 对给定的idx（或者是indices的hierarchy组合体）, 返回token（或者是tokens对应的hierarchy组合体）
    def to_tokens(self, indices):
        '''提供数字to文字映射, 对给定的idx, 返回其对应的token. 如果输入有hierarchy结构, 则以相同的结构返回'''
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.to_tokens(index) for index in indices]
    
    @property
    def token_freqs(self) -> t.List[t.Tuple]:
        return self._token_freqs