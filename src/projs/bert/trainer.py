import os
import torch
from torch import nn as nn
from torch.utils.data.dataloader import default_collate
from ...core.design.train_tools import easyTrainer





class bertPreTrainer(easyTrainer):
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
            ( _tokenID[index], _valid_lens[index], _segment[index], _mask_position[index]] ),
            ( _mlm_valid_lens[index], _mlm_label[index], _nsp_label[index] )
        经过 default_collate(batch_list), 返回 batch data:
            ( _tokenID[batch::], _valid_lens[batch::], _segment[batch::], _mask_position[batch::]] ),
            ( _mlm_valid_lens[batch::], _mlm_label[batch::], _nsp_label[batch::] )

        逐一move到cuda上
        '''

        assert hasattr(self, 'device'), \
            f"Device not set. Please set trainer's device before setting data iterators"

        def move_to_cuda(batch_list):
            (_tokenID, _valid_lens, _segment, _mask_position), (_mlm_valid_lens, _mlm_label, _nsp_label) = default_collate(batch_list)

            _tokenID = _tokenID.to(self.device)
            _valid_lens = _valid_lens.to(self.device)
            _segment = _segment.to(self.device)
            _mask_position = _mask_position.to(self.device)
            _mlm_valid_lens = _mlm_valid_lens.to(self.device)
            _mlm_label = _mlm_label.to(self.device)
            _nsp_label = _nsp_label.to(self.device)

            return (_tokenID, _valid_lens, _segment, _mask_position), (_mlm_valid_lens, _mlm_label, _nsp_label)
        
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

        # net_inputs_batch        
        # tokens: (batch_size, seq_len)int64 ot token ID. 已包含<cls>和<sep>
        # valid_lens: (batch_size,)
        # segments: (batch_size, seq_len)01 indicating seq1 & seq2 | None, None 代表当前 batch 不需要进入 NSP task
        # mask_positions: (batch_size, num_masktks) | None, None 代表当前 batch 不需要进入 MLM task

        # loss_inputs_batch
        # mlm_valid_lens (batch_size,)
        # mlm_label (batch_size, num_masktks)
        # nsp_label (batch_size,)
        mlm_valid_lens, mlm_label, nsp_label = loss_inputs_batch

        # embd_X (batch_size, max_len, num_hiddens)
        # mlm_Y_hat (batch_size, num_masktks, vocab_size)
        # nsp_Y_hat (batch_size, 2)
        embd_X, mlm_Y_hat, nsp_Y_hat = net(*net_inputs_batch)


        # get loss
        mlm_l, nsp_l = loss(mlm_Y_hat, mlm_label, mlm_valid_lens, nsp_Y_hat, nsp_label)

        return mlm_l, nsp_l, embd_X # (batch_size,), (batch_size,), (batch_size, max_len, num_hiddens)
    

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
                self.optimizer.zero_grad()

                _, _, _ = self.FP_step(self.net, self.loss, net_inputs_batch, loss_inputs_batch)

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
            self.epoch_evaluator.judge_epoch(epoch)


            for net_inputs_batch, loss_inputs_batch in self.train_iter:

                self.optimizer.zero_grad()

                mlm_l, nsp_l, embd_X = self.FP_step(self.net, self.loss, net_inputs_batch, loss_inputs_batch)
                l = mlm_l + nsp_l

                # bp
                l.sum().backward()

                if hasattr(self, 'grad_clip_val') and self.grad_clip_val is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_val)

                self.optimizer.step()

                with torch.no_grad():
                    self.epoch_evaluator.record_batch(net_inputs_batch, loss_inputs_batch, mlm_l, nsp_l)

            with torch.no_grad():
                # 如果 valid_iter 非 None, 那么在确定要 evaluate model 的 epoch, 将遍历 整个 valid_iter 得到 validation loss
                self.epoch_evaluator.evaluate_model(self.net, self.loss, self.valid_iter, self.FP_step)
                # cast metric summary
                self.epoch_evaluator.cast_metric()
        
        print('Fitting finished successfully')
