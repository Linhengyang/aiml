import torch
from torch import nn as nn
import typing as t
from torch.utils.data.dataloader import default_collate
from src.core.interface.infra_easy import easyTrainer



class transformerTrainer(easyTrainer):
    def __init__(self, net, loss, num_epochs, batch_size):
        super().__init__()
        self.net = net
        self.loss = loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def log_topology(self, logfile_path):
        '''
        日志打印网络拓扑结构到 file path: /workspace/log/[proj_name]/[XXX_topo].txt
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
    
    def set_data_iter(self, train_set, valid_set=None, test_set=None):
        assert hasattr(self, 'device'), \
            f"Device not set. Please set trainer's device before setting data iterators"
        
        def move_to_cuda(batch_list):
            src, src_valid_lens, tgt, tgt_valid_lens = default_collate(batch_list)
            src = src.to(self.device)
            src_valid_lens = src_valid_lens.to(self.device)
            tgt = tgt.to(self.device)
            tgt_valid_lens = tgt_valid_lens.to(self.device)
            return src, src_valid_lens, tgt, tgt_valid_lens
        
        self.train_iter = torch.utils.data.DataLoader(train_set, self.batch_size, True, collate_fn=move_to_cuda)
        # 是否输入 validate dataset
        if valid_set:
            self.valid_iter = torch.utils.data.DataLoader(valid_set, self.batch_size, False, collate_fn=move_to_cuda)
        else:
            self.valid_iter = None
        # 是否输入 test dataset
        if test_set:
            self.test_iter = torch.utils.data.DataLoader(test_set, self.batch_size, False, collate_fn=move_to_cuda)
        else:
            self.test_iter = None
    
    @staticmethod
    def FP_step(net:nn.Module, loss:nn.Module, net_inputs_batch, loss_inputs_batch):
        # net_inputs_batch, loss_inputs_batch 从 data_iter 中生成
        # Y_label: (batch_size, num_steps)
        # Y_valid_lens: (batch_size,)
        Y_label, Y_valid_lens = loss_inputs_batch
        Y_hat, _ = net(*net_inputs_batch) # Y_hat, tensor of logits(batch_size, num_steps, vocab_size), None
        # get loss
        l = loss(Y_hat, Y_label, Y_valid_lens)
        return l, Y_hat

    def resolve_net(self, need_resolve, bos_id:int|None=None):
        '''
        用test_data_iter的first batch对net作一次forward计算, 使得所有lazyLayer被确定(resolve).随后检查整个前向过程和loss计算(check)
        return:
            resolve且check无误 --> True;
            resolve或check有问题 --> raise AssertionError;
            不resolve或check直接训练 --> False
        '''
        if need_resolve and isinstance(bos_id, int):
            assert hasattr(self, 'test_iter') and self.test_iter, \
                f'Please first set valid test_set dataset when deploying .set_data_iter'
            self.net.train()
            # 取 inputs
            for src, src_valid_lens, tgt, tgt_valid_lens in self.test_iter:
                break
            try:
                self.optimizer.zero_grad()
                batch_size = src.size(0)
                bos = torch.empty((batch_size, 1), dtype=torch.int64, device=self.device).fill_(bos_id)
                tgt_hat, _ = self.net(src, torch.cat([bos, tgt[:, :-1]], dim=1), src_valid_lens, tgt_valid_lens)
                _ = self.loss(tgt_hat, tgt, tgt_valid_lens)
                self.net_resolved = True
                print('Net & Loss forward succeed. Net & Loss checked. Ready to fit')
            except:
                self.net_resolved = False
                raise AssertionError(
                    f'Net & Loss forward failed. Please check code'
                    )
        else:
            self.net_resolved = False
            print('Net unresolved. Net & Loss unchecked. Ready to skip init_params to fit directly')
        
        return self.net_resolved
    
    def init_params(self):
        '''customize the weights initialization behavior and initialize the resolved net'''
        assert hasattr(self, 'net_resolved') and self.net_resolved, \
            f'network unresolved. Must resolve network before applying init_params'
        def xavier_init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
        self.net.apply(xavier_init_weights) # net.apply 递归式地调用 fn 到 inner object
    
    def set_optimizer(self, lr, optim_type='AdamW', w_decay=None):
        '''set the optimizer at attribute optimizer'''
        assert type(lr) == float, 'learning rate should be a float'
        if optim_type == 'AdamW' and isinstance(w_decay, float):
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=w_decay)
        elif optim_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        else:
            raise NotImplementedError(f'optimizer {optim_type} not implemented')
    
    def set_grad_clipping(self, grad_clip_val:None|float = None):
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

    def fit(self, bos_id: int):
        assert hasattr(self, 'device'), 'device is not specified'
        assert hasattr(self, 'optimizer'), 'optimizer missing'
        assert hasattr(self, 'train_iter'), 'data_iter missing'
        assert hasattr(self, 'epoch_evaluator'), 'epoch_evaluator missing'
        for epoch in range(self.num_epochs):
            # model set to train
            self.net.train()
            # evaluator determine if this epoch to reveal train situation /  evaluate current network
            self.epoch_evaluator.judge_epoch(epoch)
            for src, src_valid_lens, tgt, tgt_valid_lens in self.train_iter:
                # zero-grad reset
                self.optimizer.zero_grad()
                batch_size = src.size(0)
                bos = torch.empty((batch_size, 1), dtype=torch.int64, device=self.device).fill_(bos_id)
                tgt_hat, _ = self.net(src, torch.cat([bos, tgt[:, :-1]], dim=1), src_valid_lens, tgt_valid_lens)
                l = self.loss(tgt_hat, tgt, tgt_valid_lens)
                # 反向传播 & 参数更新
                l.sum().backward()
                if hasattr(self, 'grad_clip_val') and self.grad_clip_val is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_val)
                self.optimizer.step()
                with torch.no_grad():
                    self.epoch_evaluator.record_batch(tgt_valid_lens, l)
            with torch.no_grad():
                # 如果 valid_iter 非 None, 那么在确定要 evaluate model 的 epoch, 将遍历 valid_iter 得到 validation loss
                self.epoch_evaluator.evaluate_model(self.net, self.loss, self.valid_iter, bos_id)
                # cast metric summary
                self.epoch_evaluator.cast_metric()
        
        print('Fitting finished successfully')
