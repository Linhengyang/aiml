import os
import torch
import math
from ...Compute.EvaluateTools import Timer, Accumulator, epochEvaluator
from ...Compute.VisualizeTools import Animator
import yaml
configs = yaml.load(open('Code/projs/cfrec/configs.yaml', 'rb'), Loader=yaml.FullLoader)
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
                S_hat, P, bu, Q, bi = net(users, items)
                S = torch.zeros(net.U, net.I, device=users.device)
                S[users, items] = scores
                l = loss(S_hat, S, P, Q, bu, bi)
                mse = torch.sum((S_hat - S).pow(2))
                self.eval_metric.add(l, mse, len(scores))

    def epoch_metric_cast(self):
        '''log file path: ../logs/cfrec/xxxx.txt'''
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