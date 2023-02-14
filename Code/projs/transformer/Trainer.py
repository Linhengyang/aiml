import os
import torch
from torch import nn as nn
from .Network import Transformer
from ...Compute import easyTrainer

class transformerTrainer(easyTrainer):
    def __init__(self, net, loss, num_epochs):
        super().__init__()
        self.net = net
        self.loss = loss
        self.num_epochs = num_epochs
        self.device = None
        self.grad_clip_val = None
    
    def log_topology(self, fname, topos_dir='/Users/lhy/studyspace/online/topos'):
        '''log the topology of the network to topos directory
        file path: /Users/lhy/studyspace/online/topos/transformer/XXXXX.txt
        '''
        with open(os.path.join(topos_dir, 'transformer', fname), 'w') as f:
            print(self.net, file=f)
    
    def set_device(self, device=None):
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            print('Using cpu as device')
            self.device = torch.device('cpu')
        self.net.to(self.device)

    @staticmethod
    def _decorate_data_iter(device, data_iter, tgt_vocab):
            if data_iter is None:
                return None
            for batch in data_iter:
                X, X_valid_len, Y, Y_valid_len = [t.to(device) for t in batch]
                bos = torch.tensor( [tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
                dec_X = torch.cat([bos, Y[:, :-1]], dim=1)
                net_inputs_batch = [X, dec_X, X_valid_len]
                loss_inputs_batch = [Y, Y_valid_len]
                yield (net_inputs_batch, loss_inputs_batch)

    def set_data_iter(self, tgt_vocab, train_data_iter, valid_data_iter, test_data_iter=None):
        assert self.device is not None, "Device not set. Please set trainer's device before setting data_iters"
        self.train_data_iter = self._decorate_data_iter(self.device, train_data_iter, tgt_vocab)
        self.valid_data_iter = self._decorate_data_iter(self.device, valid_data_iter, tgt_vocab)
        self.test_data_iter = self._decorate_data_iter(self.device, test_data_iter, tgt_vocab)

    def resolve_net(self, need_resolve=False):
        if need_resolve:
            assert self.test_data_iter is not None, 'Please set the test_data_iter when setting data iters if net resolving and checking wanted'
            self.net.train()
            for net_inputs_batch, loss_inputs_batch in self.test_data_iter:
                break
            try:
                Y_hat, _ = self.net(*net_inputs_batch)
                l = self.loss(Y_hat, *loss_inputs_batch).sum()
                del Y_hat, l
                print('Net resolved. Net & Loss checked. Ready to fit')
                return True
            except:
                raise AssertionError('Net or Loss has problems. Please check code before fit')
        print('Net unresolved. Net & Loss unchecked. Ready to skip param init and fit')
        return False
    
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
            for batches in self.train_data_iter:
                self.optimizer.zero_grad()
                net_inputs_batch, loss_inputs_batch = batches
                Y_hat, _ = self.net(*net_inputs_batch)
                l = self.loss(Y_hat, *loss_inputs_batch)
                l.sum().backward()
                if self.grad_clip_val is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_val)
                self.optimizer.step()
                with torch.no_grad():
                    pass
                    # self.evaluator()
                del net_inputs_batch, loss_inputs_batch, Y_hat, l
            # if (epoch + 1) % 100 == 0:
                # self.visualizer()
                # self.save_model()
    