import torch
from torch import nn
import math
from ...Compute.PredictTools import easyPredictor

def users_items_rating_pred(users, items, net):
    '''
    users: (batch_size,)int64
    items: (batch_size,)int64
    '''
    net.eval()
    with torch.no_grad():
        net_outputs = net(users, items)
        if len(net_outputs) == 1:
            pred_ratings = net_outputs[0]
        else:
            pred_ratings = net_outputs[0][users, items]
    return pred_ratings

def rmse(pred_ratings, truth_ratings):
    assert pred_ratings.shape == truth_ratings.shape, 'preds and truth shape mismatch'
    return math.sqrt( torch.sum((pred_ratings - truth_ratings).pow(2)) / pred_ratings.numel() )

class MovieRatingMFPredictor(easyPredictor):
    def __init__(self, device=None):
        super().__init__()
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.pred_fn = users_items_rating_pred
        self.eval_fn = rmse

    def predict(self, users, items, net):
        net.to(self.device)
        self.preds = self.pred_fn(users.to(self.device), items.to(self.device), net)
        return self.preds

    def evaluate(self, ratings):
        assert hasattr(self, 'preds'), 'pred result not found'
        self.truth = ratings.to(self.device)
        return self.eval_fn(self.preds, self.truth)
