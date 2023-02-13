import os
import torch
from torch import nn as nn
from .Network import Transformer
from ...Compute import easyTrainer

class transformerTrainer(easyTrainer):
    def __init__(self, net, loss, train_data_iter, num_epochs, device, 
                 test_data_iter=None, init_batch=None):
        super().__init__()
        self.net = net
        self.loss = loss
        self.device = device
        self.num_epochs = num_epochs
        self.train_data_iter = train_data_iter
        self.test_data_iter = test_data_iter
        self.init_batch = init_batch
    
    def log_topology(self, fname, topos_dir='/Users/lhy/studyspace/online/topos'):
        '''log the topology of the network to topos directory
        file path: /Users/lhy/studyspace/online/topos/transformer/XXXXX.txt
        '''
        with open(os.path.join(topos_dir, 'transformer', fname), 'w') as f:
            print(self.net, file=f)
    
    def set_device(self, device=None):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')

    def set_data_iter(self, train_data_iter, test_data_iter=None):
        def inner_iter(data_iter):
            if data_iter is None:
                return None
            for batch in data_iter:
                # set device
                yield batch
        self.train_data_iter = inner_iter(train_data_iter)
        self.test_data_iter = inner_iter(test_data_iter)

    def resolve_net(self):
        '''
        only when init_batch is a train_iter-like data generator, this function can be executed
        '''
        assert self.init_batch is not None, "A data iter as init_batch is needed to resolve net's lazy layer"
        self.net.train()
        for X, X_valid_len, Y, Y_valid_len in self.init_batch:
            break
        self.net(X, Y, X_valid_len)
    
    def init_params(self):
        '''
        customize the weights initialization behavior and initialize the net
        '''
        def xavier_init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
        self.net.apply(xavier_init_weights)
    
    def set_optimizer(self, lr, optim_type='adam'):
        '''
        set the optimizer at attribute optimizer
        '''
        assert type(lr) == float, 'learning rate should be a float'
        if optim_type == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
    
    def set_grad_clipping(self, grad_clip_val=None):
        '''
        set the grad_clip_val at attribute grad_clip_val if input a float
        '''
        self.grad_clip_val = grad_clip_val

    def fit(self):
        assert self.optimizer is not None, 'No optimizer detected in trainer'
        self.net.train()
        for epoch in range(self.num_epochs):
            for batch in self.train_data_iter:
                self.optimizer.zero_grad()
                ## get data and compute loss
                X, X_valid_len, Y, Y_valid_len = batch
                Y_hat, _ = self.net(X, Y, X_valid_len)
                l = self.loss(Y_hat, Y, Y_valid_len)
                ## bp/update params/evaluate model
                l.sum().backward()
                if self.grad_clip_val is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_val)
                self.optimizer.step()
                with torch.no_grad():
                    self.evaluator()
            if (epoch + 1) % 100 == 0:
                self.visualizer()
    