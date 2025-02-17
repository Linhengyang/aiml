import warnings
warnings.filterwarnings("ignore")
from Code.Utils.Text.BytePairEncoding import get_BPE_symbols
import re
if __name__ == "__main__":
    '''
    test code in this test.py file. After successful tests, code will be moved to _jobs.py under proj_name
    '''
    tail_token: str = '</z>'
    text = "?low   low low lower;. lower.   \n  lower high higher\thigh high high big)bigger bigger,bigger.."
    symbols = get_BPE_symbols(text, tail_token, merge_times=10)

    print(symbols)