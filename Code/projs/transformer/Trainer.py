import os
import torch
from torch import nn as nn
from torch.utils.data.dataloader import default_collate
from ...Compute.Trainers import easyTrainer
from ...Compute.EvaluateTools import Timer, Accumulator

class transformerTrainer(easyTrainer):
    def __init__(self, net, loss, num_epochs, batch_size):
        super().__init__()
        self.net = net
        self.loss = loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.eval_cnts = 10
    
    def log_topology(self, fname, logs_dir='/Users/lhy/studyspace/online/logs'):
        '''file path: /Users/lhy/studyspace/online/logs/transformer/XXXXX.txt'''
        with open(os.path.join(logs_dir, 'transformer', fname), 'w') as f:
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
        ''' 输入train_set(必须), valid_set(可选), test_set(可选, 用于resolve网络), 依次得到train_iter, valid_iter, test_iter'''
        assert hasattr(self, 'device'), "Device not set. Please set trainer's device before setting data_iters"
        def transfer(x):
            result = []
            for x_ in default_collate(x):
                res_ = tuple()
                for t in x_:
                    res_ += (t.to(self.device),)
                result.append(res_)
            return tuple(result)
        self.train_iter = torch.utils.data.DataLoader(train_set, self.batch_size, True, collate_fn=transfer)
        if valid_set:
            self.valid_iter = torch.utils.data.DataLoader(valid_set, self.batch_size, False, collate_fn=transfer)
        if test_set:
            self.test_iter = torch.utils.data.DataLoader(test_set, self.batch_size, False, collate_fn=transfer)
    
    def resolve_net(self, need_resolve=False):
        '''
        用test_data_iter的first batch对net作一次forward计算, 使得所有lazyLayer被确定(resolve).随后检查整个前向过程和loss计算(check).
        resolve且check无误 --> True; resolve或check有问题 --> raise AssertionError; 不resolve或check直接训练 --> False
        '''
        if need_resolve:
            assert hasattr(self, 'test_iter'), 'Please input test_set when deploying .set_data_iter()'
            self.net.train()
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
        self.net.apply(xavier_init_weights)
    
    def set_optimizer(self, lr, optim_type='adam'):
        '''set the optimizer at attribute optimizer'''
        assert type(lr) == float, 'learning rate should be a float'
        if optim_type == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
    
    def set_grad_clipping(self, grad_clip_val=None):
        '''set the grad_clip_val at attribute grad_clip_val if input a float'''
        self.grad_clip_val = grad_clip_val

    def evaluator(self, l, net_inputs_batch):
        '''
        train过程中的评价器, 记录每一个epoch的:
            1. train loss(sum of loss among train batch/valid hat nums among train batch) 
            2. train accuracy(sum of correct preds among train batch/nums among train batch)
            3. valid loss(sum of loss among valid batch/valid hat nums among valid batch)(可选)
            4. valid accuracy(sum of correct preds among valid batch/nums among valid batch)
        其中前两个train的指标, 在batch内部记录得到; 后两个valid的指标, 在当前epoch对valid_data作用net得到
        '''
        raise NotImplementedError

    def visualizer(self, *args, **kwargs):
        '''可视化train过程'''
        raise NotImplementedError
    
    def save_model(self, fname, models_dir='/Users/lhy/studyspace/online/model'):
        '''save the model to model directory'''
        save_path = os.path.join(models_dir, 'transformer', fname)
        torch.save(self.net.state_dict(), save_path)

    def fit(self):
        assert hasattr(self, 'optimizer'), 'Optimizer is not defined'
        assert hasattr(self, 'train_iter'), 'Data_iter is not defined'
        self.net.train()
        for epoch in range(self.num_epochs):
            record_flag = (epoch + 1) % (self.num_epochs // self.eval_cnts) == 0
            if record_flag:
                timer = Timer()
                metric = Accumulator(2)
            
            for net_inputs_batch, loss_inputs_batch in self.train_iter:
                self.optimizer.zero_grad()
                Y_hat, _ = self.net(*net_inputs_batch)
                l = self.loss(Y_hat, *loss_inputs_batch)
                l.sum().backward()
                if hasattr(self, 'grad_clip_val') and self.grad_clip_val is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_val)
                self.optimizer.step()

                with torch.no_grad():
                    if record_flag:
                        # batch loss(sum of loss of sample), batch num_tokens(sum of Y_valid_lens pf sample)
                        metric.add( l.sum(), loss_inputs_batch[1].sum() )

                del net_inputs_batch, loss_inputs_batch, Y_hat, l
            
            if record_flag:
                cur_epoch_time = timer.stop()
                print(f' train epoch {epoch+1}: loss {metric[0] / metric[1]:.3f}'
                      f' speed {metric[1] / cur_epoch_time:.1f} tokens/sec on {str(self.device)}'
                      f' expected remaining time {cur_epoch_time * (self.num_epochs-epoch-1) / 60:.1f} mins')
        print('Fitting finished successfully')