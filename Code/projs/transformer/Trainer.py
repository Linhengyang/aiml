import os
from .Network import Transformer
from ...Compute import Trainer

class transformerTrainer(Trainer):
    def __init__(self, net, loss, train_data_iter, grad_clip_val, num_epochs, device):
        super().__init__()
        self.net = net
        self.loss = loss
        self.grad_clip_val = grad_clip_val
        self.device = device
        self.num_epochs = num_epochs
        self.train_data_iter = train_data_iter
    
    def topo_logger(self, fname, topos_dir='/Users/lhy/studyspace/online/topos'):
        '''log the topology of the network to topos directory'''
        with open(os.path.join(topos_dir, 'transformer', fname), 'w') as f:
            print(self.net, file=f)
    
    def net_resolver(self):
        self.net.train()
        for X, X_valid_len, Y, Y_valid_len in self.train_data_iter:
            break
        self.net(X, Y, X_valid_len)