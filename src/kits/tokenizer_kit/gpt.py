# functions to transfer vanilla BPE tokenizer to GPT-style adapted tokenizer

from ...core.utils.text.tokenizer import boostBBPETokenizer, merge_pair
import json
import typing as t



def bytes_to_unicode() -> dict[int, str]:
    '''
    GPT 使用的 tokenizer 为了保证字典/合并信息可正常以 json 形式渲染打印, 对 0-255 的原始 byte 作了一个 byte 到 unicode 字符的 双射映射:
    按序如下
    ASCII可打印字符            33   -   126 -->  unicode char 33 -126 (恒等映射)
    Latin1可打印字符           161  -   172 -->  unicode char 161-172 (恒等映射)
    Latin1可打印字符            174 -   255 -->  unicode char 174-255 (恒等映射)
    控制字符和单空格  (共33个)   0   -   32  -->  unicode char 256-288
    DEL和其他控制字符 (共34个)  127  -   160 -->  unicode char 289-322
    soft-hyphen字符 (共1个)        173      -->  unicode char 323
    
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



class gpt2Tokenizer(boostBBPETokenizer):

    GPT2_TOKENIZER_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    B2U: dict = bytes_to_unicode() # byte-integer (0-255) --> unicode char (256个可打印字符)

    U2B: dict = {v:k for k, v in B2U.items()} # unicode char (256个可打印字符) --> byte-integer (0-255)

    '''
    original BBPE Tokenizer 包含 _merge_ranks: L_ID, R_ID --> merged_ID 用于编码, 和 _vocab/special_tokens: ID --> bytes 用于解码
    gpt BBPE TOkenizer 包含 merges: L_b2u_mapped_chars, R_b2u_mapped_chars --> rank, 和 encoder: b2u_mapped_chars --> ID. encoder 包含 special_tokens
    使用 merges 用于编码(改写 original BBPE 的 _encode_chunk), 使用 encoder 构建 decoder: ID --> b2u_mapped_chars 用于解码

    original 的 _merge_ranks / gpt 的 merges 是等价的: 都是 bytes merge by frequency 的结果. 故 _vocab+special_tokens / encoder 在 ID >= 256 部分也等价
    但是 _vocab+special_tokens / encoder 在 ID 0-255 部分不等价: 前者在 ID 0-255 是原始 byte_integer <--> byte, 而后者在 ID 0-255 有重排序, 具体举例:
    对于 ID = 0, _vocab 对应 byte chr(0), 而 encoder 对应 char !

    不同的 ID-bytes(chars)映射关系, 对应了不同的 tokenize 结果. 既然本class是 gpt2tokenizer, 那么 tokenize 结果肯定要看齐 gpt2 而不是 original
    方案一: load gpt2_tokenizer, 转换 merges / encoder 到 original 的 _merge_ranks / _vocab, 然后使用 original 的 编解码 方法, 然后再将结果中 ID
            0-255 部分按照 gpt2 encoder 的顺序重新映射.
    方案二(采用): load gpt2_tokenizer 后, 改写 新的 编解码 方法, 使用 merges / encoder 直接生成结果.
    '''

    def __init__(self):
        super().__init__(name='gpt2_tok', buffer_dir='.', pat_str=self.GPT2_TOKENIZER_REGEX, special_marks=[])
    

    def to_doc(self, fpath, mode='json'):
        if mode == 'json':
            entity: dict = {}
            entity['version'] = '1.0'
            entity['truncation'] = None
            entity['padding'] = None
            entity['added_tokens'] = [] #TODO
            entity['normalizer'] = None
            entity['pre_tokenizer'] = {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': True}
            entity['post_processor'] = {'type': 'ByteLevel', 'add_prefix_space': True, 'trim_offsets': False}
            entity['decoder'] = {'type': 'ByteLevel', 'add_prefix_space': True, 'trim_offsets': True}
            entity['model'] = {}
            entity['model']['dropout'] = None
            entity['model']['unk_token'] = None
            entity['model']['continuing_subword_prefix'] = ''
            entity['model']['dropoend_of_word_suffixut'] = ''
            entity['model']['fuse_unk'] = False
            entity['model']['vocab'] = {} #TODO
            entity['model']['merges'] = [] #TODO

    

    def from_doc(self, fpath, mode='json'):
        if mode == 'json':
            with open(fpath) as f:
                entity = json.load(f)

            self.encoder = entity['model']['vocab'] # b2u token(str) --> ID(int)
            self.decoder = {v: k for k, v in self.encoder.items()} # ID(int) --> b2u token(str)
            self.merges = {pair: i+256 for i, pair in enumerate(entity['model']['merges'])} # 'L_b2u_token R_b2u_token'(str) --> rank(int)

            self.explicit_n_vocab = len(self.encoder)

            # 读取 special marks
            self._special_marks = [item['content'] for item in entity['added_tokens'] if item['special']]
            # special mark str --> special token id
            self.special_tokens = {item['content']: int(item['id']) for item in entity['added_tokens'] if item['special']}
            # special token id --> special mark str
            self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
    

    # 重写 _encode_chunk 即可实现 encode
    # _encode_chunk 把 byte-ints 经 b2u 映射后, 得到 list of uchars(str) 上, 再根据 merges 尽可能合并成 tokens 后, 用 encoder 映射到 token ID
    def _encode_chunk(self, tokens:t.List[int]) -> t.List[int]:
        '''
        对 chunk(tokens) 作持续的 merge, 直至 无可merge
        '''
        tokens = [self.B2U[idx] for idx in tokens] # list of uchars(str)
        while len(tokens) > 1: # 在循环体内不断更新 tokens
            pairs: str = [' '.join(pair) for pair in zip(tokens, tokens[1:])] # 取出所有 pair, 用空格 concat
            # 取出 合并 rank 最小的 pair. 当无可合并时, 即 pairs 中无一存在于 merges, 那么 min_rank_pair 不存在与 merges, 该跳出合并循环
            min_rank_pair: str = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
            if min_rank_pair not in self.merges:
                break
            # 更新 tokens
            tokens = merge_pair(tokens, min_rank_pair.split(' '), min_rank_pair.replace(' ', ''))
        
        return [self.encoder[token] for token in tokens]


    # 重写 decode
    def decode(self, tokens:t.List[int], errors: str = "replace") -> str:
        assert hasattr(self, 'decoder')

        # 根据 decoder 将 int ID 转换为 b2u chars, 再 逆映射回 bytes
        parts = []
        for idx in tokens:
            # decoder 是 包含 inverse_special_tokens 的, 故先判断 inverse_special_tokens
            if idx in self.inverse_special_tokens:
                parts.append( self.inverse_special_tokens[idx].encode('utf-8') ) # append bytes
            elif idx in self.decoder:
                u_chars: str = self.decoder[idx]
                parts.append(bytes([self.U2B[c] for c in u_chars])) # append bytes
            else:
                raise ValueError(f'invalid index {idx} out-of-vocab')
        
        concat_bytes = b''.join( parts )
        # 容错：错误的字节序列使用一个 replacement char 来替代
        return concat_bytes.decode('utf-8', errors=errors)