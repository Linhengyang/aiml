import os
import Code.projs.transformer.DataLoad_seq2seq as py_doc
from Config import base_data_dpath, seq2seq_dname, eng2fra_fname

if __name__ == "__main__":
    eng2fra = os.path.join(base_data_dpath, seq2seq_dname, eng2fra_fname)
    train_iter, src_vocab, tgt_vocab = py_doc.data_loader_seq2seq(path=eng2fra, batch_size=2, num_steps=8, num_examples=600)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X)
        print('valid lengths for X:', X_valid_len)
        print('Y:', Y)
        print('valid lengths for Y:', Y_valid_len)
        print(X.dtype, Y.dtype)
        break