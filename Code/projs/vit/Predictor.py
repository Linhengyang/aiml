import torch
from torch import nn
from ...Compute.PredictTools import easyPredictor
from ...Compute.EvaluateTools import accuracy
from .Dataset import decode_idx3_ubyte, decode_idx1_ubyte
from ...Utils.Image.Display import display_images_with_labels



def imageTensor_batch_predict(net, imgTensor_batch, device):
    '''
    net:
    imgTensor_batch: image batch tensors, shape(batch_size, num_channels, height, width)
    device: 手动管理设备. 应该和 net 的设备一致

    output: @ CPU
    predicted class list(int), predicted scores list(float)
    '''
    input_tensor = imgTensor_batch.to(device)
    net.to(device)

    net.eval()
    with torch.no_grad():
        Y_hat = net(input_tensor) # Y_hat shape:(batch_size, num_classes)
        pred_result = nn.Softmax(dim=1)(Y_hat).max(dim=1)
    
    pred_indices, pred_scores = pred_result.indices.tolist(), pred_result.values.tolist()

    # return [pred_cls1, .., pred_clsN], [pred_score1, .., pred_scoreN]
    return pred_indices, pred_scores



class fmnistClassifier(easyPredictor):

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    def __init__(self, net, device=None):
        super().__init__()
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            self.device = torch.device('cpu')
        print(f"use device {self.device} to infer")

        self.pred_fn = imageTensor_batch_predict
        self.eval_fn = accuracy

        self.net = net


    def predict(self, imgdata_fpath):
        input = decode_idx3_ubyte(imgdata_fpath).type(torch.float32) #tensor shape:(numImgs, 1, numRows, numCols)

        # pred_indices: list of int, pred_scores: list of float
        self.pred_indices, self._pred_scores = self.pred_fn(self.net, input, self.device)

        pred_labels = [ self.classes[i] for i in self.pred_indices ]

        display_images_with_labels(input, pred_labels)
    

    def evaluate(self, imglabel_fpath):
        assert hasattr(self, 'pred_indices'), f'predicted class indices not found'

        truths = decode_idx1_ubyte(imglabel_fpath).type(torch.int64) #tensor shape: (numImgs, )
        preds = torch.tensor( self.pred_indices, dtype=torch.int64 ) #tensor shape: (numImgs, )

        return self.eval_fn(preds, truths) / truths.size(0)


    @property
    def pred_scores(self):
        return self._pred_scores
