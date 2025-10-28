import torch
import typing as t
import pandas as pd
import random
from torch import nn
from ...core.design.dl_outline import easyPredictor
from ...core.base.compute.evaluate_tools import accuracy
from ...core.utils.image.display import display_images_with_labels
from ...core.utils.image.mnist import decode_idx3_ubyte, decode_idx1_ubyte



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
    
    pred_classes, pred_scores = pred_result.indices.tolist(), pred_result.values.tolist()

    # return [pred_cls1, .., pred_clsN], [pred_score1, .., pred_scoreN]
    return pred_classes, pred_scores



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



    def predict(self, imgdata_fpath, select_size, view=False):

        sample_imgTensor = decode_idx3_ubyte(imgdata_fpath).type(torch.float32) # tensor shape:(numImgs, 1, numRows, numCols)

        sample_size = sample_imgTensor.size(0)

        assert select_size <= sample_size, \
            f'select number to predict must be no larger than total sample size.\
              now select size {select_size} exceeds sample size {sample_size}'
        

        # 随机选择 select_size 张图片tensor
        sample_indices = list(range(sample_size))
        random.shuffle(sample_indices)

        self._select_indices = sample_indices[:select_size] # tensor shape:(select_size,)
        self._input_imgTensor = sample_imgTensor[self._select_indices] # tensor shape:(select_size, 1, numRows, numCols)



        # pred_indices: list of int, pred_scores: list of float   both with length select_size
        self.pred_classes, self._pred_scores = self.pred_fn(self.net, self._input_imgTensor, self.device)

        pred_labels = [ self.classes[i] for i in self.pred_classes ]
        
        if view:
            print("display prediction")
            display_images_with_labels(self._input_imgTensor, pred_labels)
    


    def evaluate(self, imglabel_fpath, view=True):
        assert hasattr(self, 'pred_classes'), f'predicted class not found'

        sample_clsTensor = decode_idx1_ubyte(imglabel_fpath).type(torch.int64) # tensor shape: (numImgs, )
        input_imgclsTensor = sample_clsTensor[self._select_indices] # tensor shape: (select_size, )

        pred_imgclsTensor = torch.tensor( self.pred_classes, dtype=torch.int64 ) # tensor shape: (select_size, )

        if view:
            print("display evaluation")
            truth_labels = [ self.classes[i] for i in input_imgclsTensor ]
            pred_labels = [ self.classes[i] for i in pred_imgclsTensor ]
            correct = [p == t for p, t in zip(truth_labels, pred_labels)]

            legend_df = pd.DataFrame({'pred':pred_labels, 'truth':truth_labels, 'is_right':correct})

            display_images_with_labels(self._input_imgTensor, legend_df)


        return self.eval_fn(pred_imgclsTensor, input_imgclsTensor) / input_imgclsTensor.size(0)
    





    @property
    def pred_scores(self) -> t.List[float]:
        return self._pred_scores
