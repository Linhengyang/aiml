# functions to transfer vanilla BPE tokenizer to GPT-style adapted tokenizer
from ...core.utils.text.tokenizer import boostBBPETokenizer
import json

class gpt2Tokenizer(boostBBPETokenizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_json(self, mode='gpt2'):
        if mode == 'gpt2':
            pass

    def from_json(self, mode='gpt2'):
        if mode == 'gpt2':
            pass
    
    @staticmethod
    def bytes_to_unicode() -> dict[bytes, str]:
        '''
        GPT 使用的 tokenizer 为了保证 merges 和 vocab 可正常以 json 形式渲染打印, 对 0-255 的原始 byte 作了一个 byte 到 unicode 字符的 双射映射:
        控制字符和单空格  (共33个)   0   -   32  -->  unicode char 256-288
        ASCII可打印字符            33   -   126 -->  unicode char 33 -126 (恒等映射)
        DEL和其他控制字符 (共34个)  127  -   160 -->  unicode char 289-322
        Latin1可打印字符           161  -   172 -->  unicode char 161-172 (恒等映射)
        soft-hyphen字符 (共1个)        173      -->  unicode char 323
        Latin1可打印字符            174 -   255 -->  unicode char 174-255 (恒等映射)
        
        即 68 个{控制字符 + 空格}, 映射成 256-323 对应的 unicode 字符, 188 个可打印字符, 映射成 byte 值 对应的 unicode 字符(等价于原byte字符)
        这个双射的作用, 是使得 0-255 所有256个 byte 都可以打印, 方便把 bytes sequence 落盘成可阅读的文本json.
        '''
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]

        n = 0
        for b in range(256):
            if b not in bs: # b 对应不可打印字符
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(c) for c in cs]
        return dict(zip(bs, cs))
    