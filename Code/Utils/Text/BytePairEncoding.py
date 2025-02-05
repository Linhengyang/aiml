# BytePairEncoding.py

# a learner, learning from corpus to induce a vocaburary

# function byte-pair-encoding(corpus C, numbers of merges k) --> vocab V
# init V <-- all unique characters in C
# repear k times:
#   tok_L, tok_R <-- most frequent pair of adjacent tokens in C
#   tok_new <-- tok_L + tok_R
#   V <-- V + tok_new
#   update corpus: replace all occurrence of tok_L, tok_R in C with tok_new
# return V

corpus = "low "





# a segmenter, tokenize a raw sentences