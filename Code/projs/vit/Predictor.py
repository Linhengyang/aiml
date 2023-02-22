import torch
from torch import nn
import math
from ...Compute.PredictTools import easyPredictor
from ...Compute.EvaluateTools import accuracy

class fmnistClassifier(easyPredictor):
    def __init__(self):
        raise NotImplementedError
    def predict(self):
        raise NotImplementedError
    def evaluate(self):
        raise NotImplementedError

    @property
    def pred_scores(self):
        raise NotImplementedError
