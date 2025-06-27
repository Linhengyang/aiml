import os
import warnings
warnings.filterwarnings("ignore")
from Code.Utils.Text.Tokenizer import BBPETokenizer, ENDOFTEXT

if __name__ == "__main__":
    testTokenizer = BBPETokenizer(name="test2", special_marks=[ENDOFTEXT], explicit_n_vocab=262)
    testTokenizer.train_bpe('aaYaaaYaz ', verbose=True)
    testTokenizer.view(tmpsave_path='.')
    print(testTokenizer.vocab_size)