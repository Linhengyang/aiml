import torch
from torch import nn as nn
from torch.utils.data.dataloader import default_collate
from ...core.base.compute.train_tools import easyTrainer



class vitTrainer(easyTrainer):
    def __init__(self, net, loss, num_epochs, batch_size):
        super().__init__()
        self.net = net
        self.loss = loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size


    def log_topology(self, logfile_path):
        '''
        日志打印网络拓扑结构
        file path: /workspace/logs/[proj_name]/[XXX_topo].txt
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
        ''' 
        输入train_set(必须), valid_set(可选), test_set(可选, 用于resolve网络), 依次得到train_iter, valid_iter, test_iter

        原dataset中的 __getitem__ 返回 datapoint of index:
            _imgTensor[index], _labelTensor[index]
            
        经过 default_collate(batch_list), 返回 batch data:
            _imgTensor[batch::], _labelTensor[batch::]
        
        逐一move到cuda上
        '''

        assert hasattr(self, 'device'), \
            f"Device not set. Please set trainer's device before setting data_iters"
        
        def move_to_cuda(batch_list):
            data_batch, label_batch = default_collate(batch_list)
            data_batch = data_batch.to(self.device)
            label_batch = label_batch.to(self.device)

            return data_batch, label_batch
        
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
    

    def resolve_net(self, need_resolve=False):
        '''
        用test_data_iter的first batch对net作一次forward计算, 使得所有lazyLayer被确定(resolve).随后检查整个前向过程和loss计算(check)
        return:
            resolve且check无误 --> True;
            resolve或check有问题 --> raise AssertionError;
            不resolve或check直接训练 --> False
        '''
        if need_resolve:
            assert hasattr(self, 'test_iter') and self.test_iter, \
                f'Please first set valid test_set dataset when deploying .set_data_iter'
            
            self.net.train()

            # 取 inputs
            for X, y in self.test_iter:
                break
            try:
                self.optimizer.zero_grad()
                Y_hat = self.net(X)

                # get loss
                l = self.loss(Y_hat, y).sum()
                
                del Y_hat, l

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


            for X, y in self.train_iter:
                self.optimizer.zero_grad()
                Y_hat = self.net(X)

                # get loss
                l = self.loss(Y_hat, y)

                # bp
                l.sum().backward()

                if hasattr(self, 'grad_clip_val') and self.grad_clip_val is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_val)

                self.optimizer.step()

                with torch.no_grad():
                    self.epoch_evaluator.batch_record(X, y, Y_hat, l)

            with torch.no_grad():
                # 如果 valid_iter 非 None, 那么在确定要 evaluate model 的 epoch, 将遍历 部分 valid_iter 的batch 得到 validation loss
                self.epoch_evaluator.evaluate_model(self.net, self.loss, self.valid_iter, num_batches=10)
                # cast metric summary
                self.epoch_evaluator.epoch_metric_cast()
        
        print('Fitting finished successfully')
