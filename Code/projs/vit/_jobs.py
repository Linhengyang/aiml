import os
import warnings
warnings.filterwarnings("ignore")
import torch
from .Dataset import seq2seqDataset
from .Network import TransformerEncoder, TransformerDecoder, Transformer
from ...Loss.MaskedCELoss import MaskedSoftmaxCELoss
from .Trainer import transformerTrainer
from .Evaluator import transformerEpochEvaluator
from .Predictor import sentenceTranslator
import yaml
configs = yaml.load(open('Code/projs/transformer/configs.yaml', 'rb'), Loader=yaml.FullLoader)
local_model_save_dir = configs['local_model_save_dir']


def train_job():
    pass

def infer_job():
    pass