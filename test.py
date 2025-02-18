import warnings
warnings.filterwarnings("ignore")
from Code.Utils.Text.BytePairEncoding import get_BPE_symbols
import re
if __name__ == "__main__":
    '''
    test code in this test.py file. After successful tests, code will be moved to _jobs.py under proj_name
    '''
    tail_token: str = '</z>'
    text = "fast fast fast fast faster faster faster tall tall tall tall tall taller taller taller taller"
    symbols = get_BPE_symbols(text, tail_token, merge_times=10)

    print(symbols)