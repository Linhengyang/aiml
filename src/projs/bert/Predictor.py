import torch
import numpy as np
from ...core.base.compute.predict_tools import easyPredictor
from ...core.Utils.common.seq_operation import truncate_pad








def get_bert_embedding(
        net, vocab, max_len, tokens, tokens_next, device,
        cls_token='<cls>', eos_token='<sep>', pad_tokne='<pad>',
        *args, **kwargs):
    
    # add <cls> before tokens, add <sep> after tokens & tokens_next if any
    # produce segments to indicate 0 for tokens(including <cls> and first <sep>), 1 for tokens_next(including last <sep>)

    # tokens, segments = _concate_tokenlist_pair(tokens, tokens_next, cls_token, eos_token)
    tokens = [cls_token] + tokens + [eos_token]
    segments = [0] * len(tokens)

    if tokens_next is not None:
        tokens += tokens_next + [eos_token]
        segments += [1]*(len(tokens_next) + 1)

    valid_lens = torch.tensor([len(tokens)], dtype=torch.int32) #(1,)

    tokens = truncate_pad(tokens, max_len, pad_tokne) # token list with length max_len
    input = torch.tensor([ vocab[tokens] ], dtype=torch.int64, device=device) #(1, max_len)

    if tokens_next:
        segments = truncate_pad(segments, max_len, 0) # 0-1 list with length max_len
        segments = torch.tensor([ segments ], dtype=torch.int64, device=device) #(1, max_len)
    else:
        segments = None

    # net input: 
    # tokens: (batch_size, max_len)int64 ot token ID. 已包含 seq1前面的<cls>，和seq1/2之间 seq2后面的<sep>
    # valid_lens: (batch_size,) 非 <pad> 的token数量

    # segments: (batch_size, max_len)01 indicating seq1 & seq2 | None, None 代表当前 batch 不需要进入 NSP task
    # mask_positions: (batch_size, num_masktks) | None, None 代表当前 batch 不需要进入 MLM task
    with torch.no_grad():
        embd_, _, _ = net(input, valid_lens, segments, None) # (batch_size, max_len, num_hiddens)


    return embd_.squeeze(0).cpu().numpy() # (max_len, num_hiddens) move to cpu & numpy










class tokensEncoder(easyPredictor):
    def __init__(self,
                 vocab, # 语言的词汇表
                 net, max_len, # 模型, 步长
                 device=None):
        
        super().__init__()

        # load src/tgt language vocab
        self._vocab = vocab

        # set device
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            self.device = torch.device('cpu')
        print(f"use device {self.device} to infer")

        # set predict function/model/max_len
        self.max_len = max_len
        self.net = net
        self.pred_fn = get_bert_embedding

        # set evaluate function
        self.eval_fn = None



    def predict(self, tokens, tokens_next=None, cls_token='<cls>', eos_token='<sep>', pad_tokne='<pad>'):

        # (max_len, num_hiddens) cpu & numpy --extract 1 to len(tokens)--> where tokens embedding

        # only extract embedding matrix for input tokens
        self.embed_array = self.pred_fn(self.net, self._vocab, self.max_len, tokens, tokens_next,
                                        self.device, cls_token, eos_token, pad_tokne)[1:len(tokens)+1]
        
        return self.embed_array
        


    def evaluate(self):
        pass
    


    @property
    def pred_scores(self):
        pass
