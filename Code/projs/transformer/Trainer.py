import os
import torch
from torch import nn as nn
from torch.utils.data.dataloader import default_collate
from ...Compute.TrainTools import easyTrainer
import yaml
configs = yaml.load(open('Code/projs/transformer/configs.yaml', 'rb'), Loader=yaml.FullLoader)
online_log_dir, online_model_save_dir, proj_name = configs['online_log_dir'], configs['online_model_save_dir'], configs['proj_name']


class transformerTrainer(easyTrainer):
    def __init__(self, net, loss, num_epochs, batch_size):
        super().__init__()
        self.net = net
        self.loss = loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def log_topology(self, fname, logs_dir=online_log_dir):
        '''file path: online_log_dir/transformer/XXX_topo.txt'''
        with open(os.path.join(logs_dir, proj_name, fname), 'w') as f:
            print(self.net, file=f)
    
    def set_device(self, device=None):
        '''指定trainer的设备'''
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            print('Using cpu as device')
            self.device = torch.device('cpu')
            
        self.net.to(self.device)

    def set_data_iter(self, train_set, valid_set=None, test_set=None):
        ''' 
        输入train_set(必须), valid_set(可选), test_set(可选, 用于resolve网络), 依次得到train_iter, valid_iter, test_iter

        原dataset中的__getitem__ 返回 datapoint of index:
            (tuple(tensor[index] for tensor in [X, Y_frontshift1, X_valid_lens]), tuple(tensor[index] for tensor in [Y, Y_valid_lens]))
            即:
            (X[index], Y_frontshift1[index], X_valid_lens[index]), (Y[index], Y_valid_lens[index])
        经过 default_collate(batch_list), 返回 batch data:
            (tuple(tensor[batch::] for tensor in [X, Y_frontshift1, X_valid_lens]), tuple(tensor[batch::] for tensor in [Y, Y_valid_lens]))
            即:
            (X[batch::], Y_frontshift1[batch::], X_valid_lens[batch::]), (Y[batch::], Y_valid_lens[batch::])

        逐一move到cuda上
        '''

        assert hasattr(self, 'device'), f"Device not set. Please set trainer's device before setting data_iters"

        def move_to_cuda(batch_list):
            (X_batch, Y_frontshift1_batch, X_valid_lens_batch), (Y_batch, Y_valid_lens_batch) = default_collate(batch_list)

            X_batch = X_batch.to(self.device)
            Y_frontshift1_batch = Y_frontshift1_batch.to(self.device)
            X_valid_lens_batch = X_valid_lens_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            Y_valid_lens_batch = Y_valid_lens_batch.to(self.device)

            return (X_batch, Y_frontshift1_batch, X_valid_lens_batch), (Y_batch, Y_valid_lens_batch)
        
        self.train_iter = torch.utils.data.DataLoader(train_set, self.batch_size, True, collate_fn=move_to_cuda)

        if valid_set:
            self.valid_iter = torch.utils.data.DataLoader(valid_set, self.batch_size, False, collate_fn=move_to_cuda)

        if test_set:
            self.test_iter = torch.utils.data.DataLoader(test_set, self.batch_size, False, collate_fn=move_to_cuda)
    

    def resolve_net(self, need_resolve=False):
        '''
        用test_data_iter的first batch对net作一次forward计算, 使得所有lazyLayer被确定(resolve).随后检查整个前向过程和loss计算(check).
        resolve且check无误 --> True; resolve或check有问题 --> raise AssertionError; 不resolve或check直接训练 --> False
        '''
        if need_resolve:
            assert hasattr(self, 'test_iter'), 'Please input test_set when deploying .set_data_iter()'
            self.net.train()
            # 取 inputs
            for net_inputs_batch, loss_inputs_batch in self.test_iter:
                break
            try:
                Y_hat, _ = self.net(*net_inputs_batch)
                l = self.loss(Y_hat, *loss_inputs_batch).sum()
                del Y_hat, l
                print('Net resolved & logged. Net & Loss checked. Ready to fit')
                return True
            except:
                raise AssertionError('Net or Loss has problems. Please check code before fit')
        
        print('Net unresolved. Net & Loss unchecked. Ready to skip init_params and fit')
        return False
    
    def init_params(self):
        '''customize the weights initialization behavior and initialize the net'''
        def xavier_init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
        
        self.net.apply(xavier_init_weights) # net.apply 递归式地调用 fn 到 inner object
    
    def set_optimizer(self, lr, optim_type='adam'):
        '''set the optimizer at attribute optimizer'''
        assert type(lr) == float, 'learning rate should be a float'
        if optim_type == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
    
    def set_grad_clipping(self, grad_clip_val:None|float = None):
        '''set the grad_clip_val at attribute grad_clip_val if input a float'''
        self.grad_clip_val = grad_clip_val

    def set_epoch_eval(self, epoch_evaluator):
        '''set the epochEvaluator to evaluate epochs during train'''
        self.epoch_evaluator = epoch_evaluator

    def save_model(self, fname, models_dir=online_model_save_dir):
        '''save the model to online model directory'''
        save_path = os.path.join(models_dir, proj_name, fname)

        torch.save(self.net.state_dict(), save_path)

    def fit(self):
        assert hasattr(self, 'device'), 'device is not specified'
        assert hasattr(self, 'optimizer'), 'optimizer missing'
        assert hasattr(self, 'train_iter'), 'data_iter missing'
        assert hasattr(self, 'epoch_evaluator'), 'epoch_evaluator missing'

        for epoch in range(self.num_epochs):
            # model set to train
            self.net.train()
            # evaluator determine if this epoch to reveal train situation /  evaluate current network
            self.epoch_evaluator.epoch_judge(epoch)


            for net_inputs_batch, loss_inputs_batch in self.train_iter:
                # net_inputs_batch = (X_batch, Y_frontshift1_batch, X_valid_lens_batch)
                # loss_inputs_batch = (Y_batch, Y_valid_lens_batch)
                self.optimizer.zero_grad()
                Y_hat, _ = self.net(*net_inputs_batch) # Y_hat, tensor of logits(batch_size, num_steps, vocab_size), None
                l = self.loss(Y_hat, *loss_inputs_batch)
                l.sum().backward()

                if hasattr(self, 'grad_clip_val') and self.grad_clip_val is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_val)

                self.optimizer.step()

                with torch.no_grad():
                    self.epoch_evaluator.batch_record(net_inputs_batch, loss_inputs_batch, Y_hat, l)

            with torch.no_grad():
                self.epoch_evaluator.evaluate_model(self.net, self.loss, self.valid_iter)
                self.epoch_evaluator.epoch_metric_cast()
        
        print('Fitting finished successfully')
