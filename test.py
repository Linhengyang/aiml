import os
import warnings
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
from Code.projs.bert.Dataset import wikitextDataset
if __name__ == "__main__":
    '''
    test code in this test.py file. After successful tests, code will be moved to _jobs.py under proj_name
    '''
    fpath = os.path.join('../../data', 'bert/wikitext-2', 'wiki.train.tokens')
    max_len = 64
    testDS = wikitextDataset(fpath, max_len)
    data_iter = torch.utils.data.DataLoader(testDS, 3, False)
    for everything in data_iter:
        print(everything)
        break