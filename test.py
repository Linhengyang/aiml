# test.py
from Code.Utils.Text.BytePairEncoding import segment_word_BPE_greedy, get_BPE_symbols


if __name__ == "__main__":
    corpus = "asdfd asdfads dfiuo dfaishd asdjkhd siduafd"
    symbols = get_BPE_symbols(
        corpus,
        tail_token="_",
        merge_times=10
    )
    print(symbols)

    res = segment_word_BPE_greedy(
        word = "asdffd",
        symbols = symbols,
        EOW_token = "_"
    )
    print(res)