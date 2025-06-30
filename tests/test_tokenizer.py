import pytest
import tiktoken
import os


from src.core.utils.text.tokenizer import BBPETokenizer


# -----------------------------------------------------------------------------
# common test data

# a few strings to test the tokenizers on
test_strings = [
    "", # empty string
    "?", # single character
    "hello world!!!? (안녕하세요!) lol123 😉", # messy string
    r"FILE:../../../data/test/text/timemachine.txt", # FILE: is handled as a special string in unpack()
]


def unpack(text):
    # we do this because `pytest -v .` prints the arguments to console, and we don't
    # want to print the entire contents of the file, it creates a mess. So here we go.
    if text.startswith("FILE:"):
        # dirname = os.path.dirname(os.path.abspath(__file__))
        # target_file = os.path.join(dirname, text[5:])
        target_file = text[5:]
        contents = open(target_file, "r", encoding="utf-8").read()
        return contents
    else:
        return text


specials_string = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! 👋<|endofprompt|>
""".strip()


special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}


llama_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()


# -----------------------------------------------------------------------------
# tests

# test encode/decode identity for a few different strings
@pytest.mark.parametrize("tokenizer_factory", [BBPETokenizer])
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity(tokenizer_factory, text):
    text = unpack(text)
    tokenizer = tokenizer_factory(name='test', explicit_n_vocab = 261) # 256 + 5, zero-merge
    tokenizer.train_bpe(corpus='')
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert text == decoded


# test bpe basic logic
@pytest.mark.parametrize("tokenizer_factory", [BBPETokenizer])
def test_wikipedia_example(tokenizer_factory):
    """
    Quick unit test, following along the Wikipedia example:
    https://en.wikipedia.org/wiki/Byte_pair_encoding

    According to Wikipedia, running bpe on the input string:
    "aaabdaaabac"

    for 3 merges will result in string:
    "XdXac"

    where:
    X=ZY
    Y=ab
    Z=aa

    Keep in mind that for us a=97, b=98, c=99, d=100 (ASCII values)
    so Z will be 256, Y will be 257, X will be 258.

    So we expect the output list of ids to be [258, 100, 258, 97, 99]
    """
    tokenizer = tokenizer_factory(name='test', explicit_n_vocab=256+3+5)
    text = "aaabdaaabac"
    tokenizer.train_bpe(text, verbose=True)
    tokens = tokenizer.encode(text)
    assert tokens == [258, 100, 258, 97, 99]
    assert tokenizer.decode(tokens) == text



# test save/load/view
@pytest.mark.parametrize("tokenizer_factory", [BBPETokenizer])
@pytest.mark.parametrize("special_marks", [ [], list(special_tokens.keys()) ])
def test_save_load(tokenizer_factory, special_marks):
    num_specials = len(special_marks)
    # do 3 merges on "aaabdaaabac"
    tokenizer = tokenizer_factory(name='test1', special_marks=special_marks, explicit_n_vocab=256+3+num_specials)
    # test on text "aaabdaaabac"
    text = "aaabdaaabac"
    tokenizer.train_bpe(corpus=text)
    # verify that save/load work as expected
    tokens = tokenizer.encode(text)
    # save the tokenizer (TODO use a proper temporary directory)
    tokenizer.save("temp/test_tokenizer_tmp.tok")
    # re-load the tokenizer
    tokenizer = BBPETokenizer(name='reload')
    tokenizer.load("temp/test_tokenizer_tmp.tok")
    # verify that decode(encode(x)) == x
    assert tokenizer.decode(tokens) == text
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.encode(text) == tokens
    # delete the temporary files
    for file in ["temp/test_tokenizer_tmp.tok"]:
        os.remove(file)




# test save/load
@pytest.mark.parametrize("tokenizer_factory", [BBPETokenizer])
@pytest.mark.parametrize("special_marks", [ [], list(special_tokens.keys()) ])
def test_complicated_text(tokenizer_factory, special_marks):
    num_specials = len(special_marks)
    tokenizer = tokenizer_factory(name='llama', special_marks=special_marks)
    # test on llama_text, with 495 merges
    text = llama_text
    tokenizer.train_bpe(corpus=text, num_merges=495)
    # verify the vocab_size
    assert tokenizer.vocab_size == 495+num_specials+256
    # verify that save/load work as expected
    tokens = tokenizer.encode(text, 'all')
    # save the tokenizer (use a proper temporary directory)
    tokenizer.save("temp/test_llama.tok")
    # re-load the tokenizer
    tokenizer = BBPETokenizer(name='reload')
    tokenizer.load("temp/test_llama.tok")
    # verify that reload is good as well
    assert tokenizer.vocab_size == 495+num_specials+256
    assert tokenizer.decode(tokens) == text
    assert tokenizer.decode(tokenizer.encode(text, 'all')) == text
    assert tokenizer.encode(text, 'all') == tokens





# test view
@pytest.mark.parametrize("tokenizer_factory", [BBPETokenizer])
def test_view(tokenizer_factory):
    tokenizer = tokenizer_factory(name='llama', special_marks={})
    tokenizer.load("temp/test_llama.tok")
    tokenizer.view('temp/')





# test empty .tok
@pytest.mark.parametrize("tokenizer_factory", [BBPETokenizer])
def test_empty(tokenizer_factory):
    tokenizer = tokenizer_factory(name='empty', special_marks={})
    tokenizer.load("temp/test_empty.tok")
    tokenizer.view('temp/')