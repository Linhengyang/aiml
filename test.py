import os
import warnings
import torch
warnings.filterwarnings("ignore")
from Code.projs.vit.Network import ViTEncoder

if __name__ == "__main__":
    '''
    test code in this test.py file. After successful tests, code will be moved to _jobs.py under proj_name
    '''
    num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens = 1, 4, 512, 0.3, 0.4, 512
    img_shape = (3, 28, 28)
    patch_size = (7, 7)
    vit = ViTEncoder(img_shape, patch_size, num_blks, num_heads, num_hiddens, emb_dropout, blk_dropout, mlp_num_hiddens)
    test_x = torch.randn((10, 3, 28, 28))
    print(vit(test_x).shape)