import os
import torch
import math
from ...Compute.EvaluateTools import Timer, Accumulator, epochEvaluator
from ...Compute.VisualizeTools import Animator
import yaml
configs = yaml.load(open('Code/projs/recsys/configs.yaml', 'rb'), Loader=yaml.FullLoader)
reveal_cnt_in_train, eval_cnt_in_train= configs['reveal_cnt_in_train'], configs['eval_cnt_in_train']
online_log_dir, proj_name = configs['online_log_dir'], configs['proj_name']


class mfEpochEvaluator(epochEvaluator):
    reveal_cnts = reveal_cnt_in_train # 披露train情况次数, 从train过程中收集
    eval_cnts = eval_cnt_in_train # 评价当前model, 需要validate data或infer.避免次数太多
    def __init__(self, num_epochs, log_fname, visualizer=None, scalar_names=['loss', 'rmse']):
        assert num_epochs >= max(self.reveal_cnts, self.eval_cnts), 'num_epochs must be larger than reveal cnts & eval cnts'
        assert len(scalar_names) >= 1, 'train loss is at least for evaluating train epochs'
        super().__init__()
        self.num_epochs, self.num_scalars = num_epochs, len(scalar_names)
        self.log_file = os.path.join(online_log_dir, proj_name, log_fname)
        with open(self.log_file, 'w') as f:
            print('train begin', file=f)
        self.legends = ['train_loss', 'val_loss', 'train_rmse', 'val_rmse']
        self.visual_flag =  visualizer
        if self.visual_flag:
            self.animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=self.legends)

    def epoch_judge(self, epoch):
        self.reveal_flag = (self.reveal_cnts != 0) and ( (epoch+1) % (self.num_epochs // self.reveal_cnts) == 0 or epoch == 0 )
        self.eval_flag = (self.eval_cnts != 0) and ( (epoch+1) % (self.num_epochs // self.eval_cnts) == 0 or epoch == 0 )
        if self.reveal_flag or self.eval_flag:
            self.epoch = epoch
            self.timer = Timer()
            self.reveal_metric = Accumulator(2*self.num_scalars) if self.reveal_flag else None
            self.eval_metric = Accumulator(2*self.num_scalars) if self.eval_flag else None
    
    def batch_record(self, S_hat, S, l, num_scores):
        if self.reveal_flag:
            mse = torch.sum((S_hat - S).pow(2))
            l2_penalty = l - mse
            self.reveal_metric.add(l, mse, l2_penalty, num_scores)
    
    def evaluate_model(self, net, loss, valid_iter, num_batches=None):
        if self.eval_flag:
            net.eval()
            for i, (users, items, scores) in enumerate(valid_iter):
                if num_batches and i >= num_batches:
                    break
                net_output = net(users, items)
                if len(net_output) == 1:
                    S_hat = net_output[0]
                    weight_params = [param for param_name, param in net.named_parameters() if param_name.endswith('weight')]
                else:
                    S_hat = net_output[0][users, items]
                    weight_params = net_output[1:]
                l = loss(S_hat, scores, *weight_params)
                mse = torch.sum((S_hat - scores).pow(2))
                self.eval_metric.add(l, mse, len(scores))

    def epoch_metric_cast(self):
        '''log file path: ../logs/recsys/xxxx.txt'''
        loss, eval_loss, rmse, eval_rmse, l2_pen = None, None, None, None, None
        reveal_log, eval_log = '', ''
        if self.reveal_flag:
            time_cost = self.timer.stop()
            loss = round(self.reveal_metric[0],3)
            rmse = round(math.sqrt(self.reveal_metric[1] / self.reveal_metric[3]),3)
            l2_pen = round(self.reveal_metric[2], 3)
            speed = round(self.reveal_metric[3]/time_cost)
            reveal_log = ",\t".join(['epoch: '+str(self.epoch+1), 'loss: '+str(loss), 'rmse/record: '+str(rmse), 'l2_penalty: '+str(l2_pen), 
                                     'speed(record/sec): '+str(speed), 'remain_time(min): '+str(round(time_cost*(self.num_epochs-self.epoch-1)/60))])
            with open(self.log_file, 'a+') as f:
                f.write(reveal_log+'\n')
        if self.eval_flag:
            eval_loss = round(self.eval_metric[0],3)
            eval_rmse = round(math.sqrt(self.eval_metric[1] / self.eval_metric[2]),3)
            eval_log = ",\t".join(['epoch: '+str(self.epoch+1), 'val_loss: '+str(eval_loss), 'val_rmse/record: '+str(eval_rmse)])
            with open(self.log_file, 'a+') as f:
                f.write(eval_log+'\n')
        if self.visual_flag:
            # 线条的顺序要和legends一一对应. 目前只支持最多4条线
            self.animator.add(self.epoch+1, (loss, eval_loss, rmse, eval_rmse))
        if (not self.visual_flag) and (self.reveal_flag or self.eval_flag):
            print(reveal_log + "\n" + eval_log)

class autorecEpochEvaluator(mfEpochEvaluator):
    def __init__(self, num_epochs, log_fname, visualizer=None, scalar_names=['loss', 'rmse']):
        super().__init__(num_epochs, log_fname, visualizer, scalar_names)
    
    def evaluate_model(self, net, loss, valid_iter, num_batches=None):
        if self.eval_flag:
            net.eval()
            for i, input_batch_matrix in enumerate(valid_iter):
                if num_batches and i >= num_batches:
                    break
                S_hat = net(input_batch_matrix)
                weight_params = [param for param_name, param in net.named_parameters() if param_name.endswith('weight')]
                l = loss(S_hat, input_batch_matrix, *weight_params)
                mse = torch.sum((S_hat*torch.sign(input_batch_matrix) - input_batch_matrix).pow(2))
                self.eval_metric.add(l, mse, torch.sign(input_batch_matrix).sum().item())

class fmEpochEvaluator(mfEpochEvaluator):
    def __init__(self, num_epochs, log_fname, visualizer=None, scalar_names=['loss', 'auc', 'accuracy']):
        assert num_epochs >= max(self.reveal_cnts, self.eval_cnts), 'num_epochs must be larger than reveal cnts & eval cnts'
        assert len(scalar_names) >= 1, 'train loss is at least for evaluating train epochs'
        self.num_epochs, self.num_scalars = num_epochs, len(scalar_names)
        self.log_file = os.path.join(online_log_dir, proj_name, log_fname)
        with open(self.log_file, 'w') as f:
            print('train begin', file=f)
        self.legends = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
        if self.visual_flag:
            self.animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=self.legends)

    def batch_record(self, y_hat, label, l):
        # shapes都是(batch_size, )
        if self.reveal_flag:
            preds = (y_hat > 0.5).type(label.dtype)
            self.reveal_metric.add(l.sum(), (preds == label).sum().item(), label.numel())
    
    def evaluate_model(self, net, loss, valid_iter, num_batches=None):
        if self.eval_flag:
            net.eval()
            for i, (features, label) in enumerate(valid_iter):
                if num_batches and i >= num_batches:
                    break
                y_hat = net(features)
                preds = (y_hat > 0.5).type(label.dtype)
                l = loss(y_hat, label)
                self.eval_metric.add(l.sum(), (preds == label).sum().item(), label.numel())

    def epoch_metric_cast(self):
        '''log file path: ../logs/recsys/xxxx.txt'''
        loss, eval_loss, acc, eval_acc = None, None, None, None
        reveal_log, eval_log = '', ''
        if self.reveal_flag:
            time_cost = self.timer.stop()
            loss = round(self.reveal_metric[0]/self.reveal_metric[2], 3)
            acc = round(math.sqrt(self.reveal_metric[1] / self.reveal_metric[2]),3)
            speed = round(self.reveal_metric[2]/time_cost)
            reveal_log = ",\t".join(['epoch: '+str(self.epoch+1), 'loss/row: '+str(loss), 'acc: '+str(acc),
                                     'speed(row/sec): '+str(speed), 'remain_time(min): '+str(round(time_cost*(self.num_epochs-self.epoch-1)/60))])
            with open(self.log_file, 'a+') as f:
                f.write(reveal_log+'\n')
        if self.eval_flag:
            eval_loss = round(self.eval_metric[0]/self.eval_metric[2], 3)
            eval_acc = round(math.sqrt(self.eval_metric[1] / self.eval_metric[2]), 3)
            eval_log = ",\t".join(['epoch: '+str(self.epoch+1), 'val_loss/row: '+str(eval_loss), 'val_acc: '+str(eval_acc)])
            with open(self.log_file, 'a+') as f:
                f.write(eval_log+'\n')
        if self.visual_flag:
            # 线条的顺序要和legends一一对应. 目前只支持最多4条线
            self.animator.add(self.epoch+1, (loss, eval_loss, acc, eval_acc))
        if (not self.visual_flag) and (self.reveal_flag or self.eval_flag):
            print(reveal_log + "\n" + eval_log)