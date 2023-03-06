import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import yaml
from .Dataset import skipgramDataset, cbowDataset
from .Network import skipgramNegSp, cbowNegSp
configs = yaml.load(open('Code/projs/word2vec/configs.yaml', 'rb'), Loader=yaml.FullLoader)
local_model_save_dir = configs['local_model_save_dir']
base_data_dir = configs['base_data_dir']
word2vec_dir = configs['word2vec_dir']
ptb_train_fname = configs['ptb_train_fname']

def skipgram_train_job():
    trainset = skipgramDataset(os.path.join(base_data_dir, word2vec_dir, ptb_train_fname))
    vocab = trainset.vocab
    # design net & loss
    embed_size = 100
    net = skipgramNegSp(len(vocab), embed_size)
    class WeightedSigmoidBCELoss(nn.Module):
        def __init__(self):
            super().__init__()
        def forwad(self, input, target, weight):
            # input shape: (batch_size, *)
            # target shape: (batch_size, *)
            out = nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction='none')
            return out.mean(dim=1) # (batch_size, )
    loss = WeightedSigmoidBCELoss()


def cbow_train_job():
    trainset = cbowDataset(os.path.join(base_data_dir, word2vec_dir, ptb_train_fname))
    vocab = trainset.vocab
    # design net & loss
    embed_size = 100
    net = cbowNegSp(len(vocab), embed_size)
    loss = nn.BCEWithLogitsLoss(reduction='none')
    