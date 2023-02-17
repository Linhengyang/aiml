import torch
from torch import nn
import math
from ...Compute.PredictTools import easyPredictor
from ...Compute.EvaluateTools import bleu
from .Dataset import build_tensorDataset
from ...Utils.Text.TextPreprocess import preprocess_space

class sentenceTranslator(easyPredictor):
    def __init__(self):
        raise NotImplementedError
    def predict(self):
        raise NotImplementedError
    def evaluate(self):
        raise NotImplementedError

    @property
    def pred_scores(self):
        raise NotImplementedError