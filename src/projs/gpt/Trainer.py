import os
import torch
from torch import nn as nn
from torch.utils.data.dataloader import default_collate
from ...Compute.TrainTools import easyTrainer
import yaml
configs = yaml.load(open('src/projs/gpt/configs.yaml', 'rb'), Loader=yaml.FullLoader)
online_log_dir, online_model_save_dir, proj_name = \
    configs['online_log_dir'], configs['online_model_save_dir'], configs['proj_name']


class projTrainer(easyTrainer):
    '''
    组成一个trainer 至少需要:
        network(nn.Module类) / loss function(也是nn.Module类, 输入pred和label, 输出 scalar loss) / num_epochs / batch_size
        device: 执行训练的机器, torch.device 对象
        data iterator: 一个 torch.utils.data.DataLoader 对象, 源源不断地给出 data batch (features 和 label)
        optimizer: 一个 torch.optimizer 对象, 输入 network 的 parameters, 和其他参数, 作用是可更新这些parameters
        epochEvaluator: 记录每个batch的训练情况、衡量整个模型, 并打印出训练过程中的各项metric

    steps:
        for epoch in epoch_loops:
            set network to train mode
            judge if train_reveal or model_eval would perform in this epoch

            loop among the data_iterator:
                get data_batch & label_batch in same length as batch_size
                optimizer set grad to 0
                get pred = net(data)
                get loss = loss(pred, label)
                l.sum().backward()
                optional: clip_grad
                optimizer step update
                in no_grad mode: record this batch's metric
            
            in no_grad mode: evaluate the mode / reveal the train metric
    '''
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

        经过 default_collate(batch_list), 返回 batch data:

        逐一move到cuda上
        '''

        assert hasattr(self, 'device'), \
            f"Device not set. Please set trainer's device before setting data iterators"
        
        def move_to_cuda(batch_list):
            # batch_list 是 batch_size 个 datapoint 组成的list
            # batches 是 batch_size 的 多个 tensors
            batches = default_collate(batch_list)
            batches.to(self.device)

            return batches
        

        self.train_iter = torch.utils.data.DataLoader(train_set, self.batch_size, True, collate_fn=move_to_cuda)
        if valid_set:
            self.valid_iter = torch.utils.data.DataLoader(valid_set, self.batch_size, False, collate_fn=move_to_cuda)
        if test_set:
            self.test_iter = torch.utils.data.DataLoader(test_set, self.batch_size, False, collate_fn=move_to_cuda)
    

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
            for net_inputs_batch, loss_inputs_batch in self.test_iter:
                break
            try:
                Y_hat, _ = self.net(*net_inputs_batch)
                l = self.loss(Y_hat, *loss_inputs_batch).sum()
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
        '''customize the weights initialization behavior and initialize the net'''

        assert hasattr(self, 'net_resolved') and self.net_resolved, \
            f'network unresolved. Must resolve network before applying init_params'


        def xavier_init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
        
        self.net.apply(xavier_init_weights) # net.apply 递归式地调用 fn 到 inner object
    
    
    def set_optimizer(self, lr, optim_type='adam'):
        '''set the optimizer at attribute optimizer'''
        assert type(lr) == float, 'learning rate should be a float'
        pass
    

    def set_grad_clipping(self, grad_clip_val=None):
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
        '''set the epochEvaluator to evaluate epochs during train'''
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
            pass
        
        print('Fitting finished successfully')
