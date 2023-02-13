import os
import torch
from torch import nn as nn
from .Network import Transformer
from ...Compute import Trainer

class transformerTrainer(Trainer):
    def __init__(self, net, loss, train_data_iter, num_epochs, device, 
                 test_data_iter=None, grad_clip_val=None, init_batch=None):
        super().__init__()
        self.net = net
        self.loss = loss
        self.grad_clip_val = grad_clip_val
        self.device = device
        self.num_epochs = num_epochs
        self.train_data_iter = train_data_iter
        self.test_data_iter = test_data_iter
        self.init_batch = init_batch
    
    def topo_logger(self, fname, topos_dir='/Users/lhy/studyspace/online/topos'):
        '''log the topology of the network to topos directory
        file path: /Users/lhy/studyspace/online/topos/transformer/XXXXX.txt
        '''
        with open(os.path.join(topos_dir, 'transformer', fname), 'w') as f:
            print(self.net, file=f)
    
    def net_resolver(self):
        '''
        only when init_batch is a train_iter-like data generator, this function can be executed
        '''
        assert self.init_batch is not None, "A data iter as init_batch is needed to resolve net's lazy layer"
        self.net.train()
        for X, X_valid_len, Y, Y_valid_len in self.init_batch:
            break
        self.net(X, Y, X_valid_len)
    
    def param_initializer(self):
        '''
        customize the weights initialization behavior
        '''
        def xavier_init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
        self.net.apply(xavier_init_weights)
    
    def param_optimizer(self, lr, optim_type='adam'):
        assert type(lr) == float, 'learning rate should be a float'
        if optim_type == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
    
    def grad_clipper(self):
        raise NotImplementedError