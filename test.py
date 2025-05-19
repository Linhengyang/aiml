# test.py

from Code.Utils.Text.Tokenize import segment_word_BPE_greedy

if __name__ == "__main__":
    # 测试 segment_word_BPE_greedy
    # glossary = None
    word = 'financialest'
    EOW_token = ''
    glossary = {'tokens': [EOW_token] + list(set('financialest')) + ['fin', 'finance', 'ial', 'est'], 'EOW_token':EOW_token}

    print(segment_word_BPE_greedy(word, EOW_appnd=False, glossary=glossary))
