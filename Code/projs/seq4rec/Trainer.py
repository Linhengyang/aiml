from ...Compute.TrainTools import easyTrainer
import torch
import torch.nn as nn


class seq4recTrainer(easyTrainer):
    def __init__(self, net, loss, num_epochs, batch_size):
        super().__init__()
        self.net = net
        self.loss = loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size
    
    def log_topology(self, *args, **kwargs):
        with open('../logs/seq4rec/seq4rec_net_topo.txt', 'w') as f:
            print(self.net, f)

    def set_data_iter(self, train_set):
        self.train_iter = torch.utils.data.DataLoader(train_set, self.batch_size, True)


    def resolve_net(self, need_resolve=False):
        if need_resolve:
            self.net.train()
            for X, y in self.train_iter:
                break

            try:
                y_hat = self.net(X)
                l = self.loss(y_hat, y).sum()
                del y_hat, l
                print('Net resolved & logged. Net & Loss checked. Ready to fit')
                return True
            except:
                raise AssertionError('Net or Loss has problems. Please check code before fit')
        
        print('Net unresolved. Net & Loss unchecked. Ready to skip init_params and fit')
        return False
    
    def set_optimizer(self, lr, optim_type='adam'):
        assert isinstance(lr, float), 'learning rate should be a float'
        if optim_type == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.sgd(self.net.parameters(), lr=lr)

    def set_epoch_eval(self, epoch_evaluator):
        self.epoch_evaluator = epoch_evaluator

    
    def save_model(self):
        torch.save(self.net.state_dict(), "../model/seq4rec/seq4rec.params")


    def fit(self):
        for epoch in range(self.num_epochs):
            self.net.train()

            for X, y in self.train_iter:
                self.optimizer.zero_grad()
                y_hat = self.net(X)
                l = self.loss(y_hat, y).sum()
                l.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()

                # with torch.no_grad():
                #     self.epoch_evaluator.batch_record(X, y, y_hat, l)
            
            # with torch.no_grad():
            #     self.epoch_evaluator.evaluate_model(self.net, self.loss, self.train_iter)
            #     self.epoch_evaluator.epoch_metric_cast()
        
        print('Fitting finished successfully')