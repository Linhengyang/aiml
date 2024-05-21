import torch
from torch import nn
import math
from ...Compute.PredictTools import easyPredictor


class seq4recPredictor(easyPredictor):
    def __init__(self):
        super().__init__()

    def predict(self, features, net):
        net.eval()
        self._preds_score = nn.functional.sigmoid( net(features) )
        return self._preds_score