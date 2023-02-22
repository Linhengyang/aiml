import os
from ...Compute.EvaluateTools import Timer, Accumulator, epochEvaluator
from ...Compute.VisualizeTools import Animator
import yaml
configs = yaml.load(open('Code/projs/transformer/configs.yaml', 'rb'), Loader=yaml.FullLoader)
reveal_cnt_in_train, eval_cnt_in_train= configs['reveal_cnt_in_train'], configs['eval_cnt_in_train']
online_log_dir, proj_name = configs['online_log_dir'], configs['proj_name']


class transformerEpochEvaluator(epochEvaluator):
    reveal_cnts = reveal_cnt_in_train # 披露train情况次数, 从train过程中收集
    eval_cnts = eval_cnt_in_train # 评价当前model, 需要validate data或infer.避免次数太多
    def __init__(self, num_epochs, log_fname, visualizer=None, scalar_names=['loss', ]):
        assert num_epochs >= max(self.reveal_cnts, self.eval_cnts), 'num_epochs must be larger than reveal cnts & eval cnts'
        assert len(scalar_names) >= 1, 'train loss is at least for evaluating train epochs'
        super().__init__()
        self.num_epochs, self.num_scalars = num_epochs, len(scalar_names)
        self.log_file = os.path.join(online_log_dir, proj_name, log_fname)
        with open(self.log_file, 'w') as f:
            print('train begin', file=f)
        legends = []
        for name in scalar_names:
            if self.reveal_cnts != 0:
                legends.append('train_'+name)
            if self.eval_cnts != 0:
                legends.append('valid_'+name)
        self.legends = legends
        self.visual_flag = visualizer
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
    
    def batch_record(self, net_inputs_batch, loss_inputs_batch, Y_hat, l):
        # Y_hat的shape是(batch_size, num_steps, vocab_size), l的shape是(batch_size,)
        if self.reveal_flag:
            # batch loss(sum of loss of sample), batch num_tokens(sum of Y_valid_lens pf sample)
            self.reveal_metric.add(l.sum(), loss_inputs_batch[1].sum())
    
    def evaluate_model(self, net, loss, valid_iter, num_batches=None):
        if self.eval_flag:
            for i, (net_inputs_batch, loss_inputs_batch) in enumerate(valid_iter):
                if num_batches and i >= num_batches:
                    break
                l = loss(net(*net_inputs_batch)[0], *loss_inputs_batch)
                self.eval_metric.add(l.sum(), loss_inputs_batch[1].sum())

    def epoch_metric_cast(self):
        '''log file path: ../logs/transformer/xxxx.txt'''
        loss_avg, eval_loss_avg, reveal_log, eval_log = None, None, '', ''
        if self.reveal_flag:
            time_cost = self.timer.stop()
            loss_avg, speed = round(self.reveal_metric[0]/self.reveal_metric[1],3), round(self.reveal_metric[1]/time_cost)
            reveal_log = ",\t".join(['epoch: '+str(self.epoch+1), 'train_loss(/token): '+str(loss_avg), 'speed(tokens/sec): '+str(speed),
                                     'remain_time(min): '+str(round(time_cost*(self.num_epochs-self.epoch-1)/60))])
            with open(self.log_file, 'a+') as f:
                f.write(reveal_log+'\n')
        if self.eval_flag:
            eval_loss_avg = round(self.eval_metric[0]/self.eval_metric[1],3)
            eval_log = ",\t".join(['epoch: '+str(self.epoch+1), 'val_loss(/token): '+str(eval_loss_avg)])
            with open(self.log_file, 'a+') as f:
                f.write(eval_log+'\n')
        if self.visual_flag:
            # 线条的顺序要和legends一一对应. 目前只支持最多4条线
            self.animator.add(self.epoch+1, (loss_avg, eval_loss_avg))
        if (not self.visual_flag) and (self.reveal_flag or self.eval_flag):
            print(reveal_log + "\n" + eval_log)