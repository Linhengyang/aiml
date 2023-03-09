import os
import warnings
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
from Code.projs.bert.Network import BERTencoder

if __name__ == "__main__":
    '''
    test code in this test.py file. After successful tests, code will be moved to _jobs.py under proj_name
    '''
    vocab_size, num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens = 10000, 2, 12, 768, 0.2, 1024
    batch_size, seq_len = 2, 8
    test_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    test_segments = torch.tensor([[0,0,0,1,1,1,1,1], [0,0,0,0,1,1,1,1]])
    valid_lens = torch.tensor([8, 8])
    net = BERTencoder(vocab_size, num_blks, num_heads, num_hiddens, dropout, ffn_num_hiddens, seq_len)
    out = net(test_tokens, test_segments, valid_lens)
    print(out.shape)