import torch
from torch import nn
import math
from ...Compute.PredictTools import easyPredictor
from ...Compute.EvaluateTools import accuracy

def tensor_batch_pred(img_batch, net):
    net.eval()
    with torch.no_grad():
        Y_hat = net(img_batch)
        pred_result = nn.Softmax(dim=1)(Y_hat).max(dim=1)
    return pred_result.indices, pred_result.values #shape都是(batch_size,)

class fmnistClassifier(easyPredictor):
    def __init__(self, device=None):
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.eval_fn = accuracy
        self.pred_fn = tensor_batch_pred

    def predict(self, img_batch, labels, net):
        self.truth = labels.to(self.device)
        net.to(self.device)
        self.preds, self._pred_scores = self.pred_fn(img_batch.to(self.device), net)
        return self.preds

    def evaluate(self):
        assert hasattr(self, 'preds'), 'pred result not found'
        return self.eval_fn(self.preds, self.truth)/self.truth.size(0)

    @property
    def pred_scores(self):
        return self._pred_scores
