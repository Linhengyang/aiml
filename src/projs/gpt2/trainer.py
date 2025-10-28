# trainer.py for gpt
import torch
from torch import nn as nn
from torch.utils.data.dataloader import default_collate
from ...core.design.dl_outline import easyTrainer



class gpt2Trainer(easyTrainer):
    def __init__(self, net, loss, num_epochs, batch_size):
        super().__init__()
        self.net = net
        self.loss = loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size


    def log_topology(self, logfile_path):
        '''
        log file path: /workspace/log/[proj_name]/[XXX_topo].txt
        '''
        with open(logfile_path, 'w') as f:
            print(self.net, file=f)
    
    
    def set_device(self, device=None):
        '''指定trainer的设备'''
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            self.device = torch.device('cpu')
        
        self.net.to(self.device)
        print(f'Using {device} as train device')
        
        
    def set_data_iter(self, train_set):

        assert hasattr(self, 'device'), \
            f"Device not set. Please set trainer's device before setting data iterators"
        
        def move_to_cuda(batch_list):
            (input_seqs, input_segs, labels, label_segs) = default_collate(batch_list)

            input_seqs = input_seqs.to(self.device)
            input_segs = input_segs.to(self.device)
            labels = labels.to(self.device)
            label_segs = label_segs.to(self.device)

            return (input_seqs, input_segs, labels, label_segs)
        
        self.train_iter = torch.utils.data.DataLoader(train_set, self.batch_size, True, collate_fn=move_to_cuda)


    def set_optimizer(self, lr: float, w_decay: float|None = None):
        '''set the optimizer at attribute optimizer'''
        if w_decay is None:
            w_decay = 0.01
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=w_decay)

    
    def set_grad_clipping(self, grad_clip_val:None|float = None):
        '''
        input grad_clip_val:
            None: 不裁剪梯度
            float: 裁剪梯度的 模 到 grad_clip_val
        '''
        if grad_clip_val:
            try:
                self.grad_clip_val = float(grad_clip_val)
            except ValueError as err:
                print(f'set float value of gradient clipping value failed {err}')
        else:
            self.grad_clip_val = None


    def set_epoch_eval(self, epoch_evaluator):
        '''设置 epochEvaluator 在训练过程中 披露 train 相关信息和 validation 相关信息'''
        self.epoch_evaluator = epoch_evaluator


    def save_model(self, modelfile_path, method='default'):
        '''保存模型参数 到 modelfile_path. 默认保存方式 method = 'default' .params格式'''
        if method == 'default':
            torch.save(self.net.state_dict(), modelfile_path)
        else:
            raise NotImplementedError(f'save method {method} not implemented')


    def fit(self, epoch_evaluator=None):

        for epoch in range(self.num_epochs):
            # model set to train
            self.net.train()

            if epoch_evaluator is not None:
                epoch_evaluator.judge_epoch(epoch)

            for input_seqs, input_segs, labels, label_segs in self.train_iter:
                self.optimizer.zero_grad()
                y_hat, _, _ = self.net(input_seqs, input_segs)
                l = self.loss(y_hat, input_segs, labels, label_segs)
                l.sum().backward()
                if self.grad_clip_val is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_val)
                self.optimizer.step()

                if epoch_evaluator is not None:
                    epoch_evaluator.record_batch(l.cpu())

            if epoch_evaluator is not None:
                epoch_evaluator.cast_metric()
        
        print('gpt2 pretrain finished')