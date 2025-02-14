import warnings
warnings.filterwarnings("ignore")
from Code.Utils.Text.BytePairEncoding import get_BPE_symbols
import re
if __name__ == "__main__":
    '''
    test code in this test.py file. After successful tests, code will be moved to _jobs.py under proj_name
    '''
    # tail_token: str = '</z>'
    # # text = "low low lower; lower.   \n  lower high higher\thigh high high"
    # text = 'lowest highest estimate'
    # symbols = get_BPE_symbols(text, tail_token, merge_times=2)

    # print(symbols)
    pattern = re.compile('(.</w>)|(.)')
    text = 'abcw</w> wa'
    for match in re.finditer(pattern, text):
        print(match.group(1))
        print(match.group(2))
        print("---")