import os
from ...Compute.EvaluateTools import Timer, Accumulator, epochEvaluator
from ....Config import reveal_cnt_in_train, eval_cnt_in_train

class transformerEpochEvaluator(epochEvaluator):
    reveal_cnts = reveal_cnt_in_train # 披露train情况次数, 从train过程中收集
    eval_cnts = eval_cnt_in_train # 评价当前model, 需要validate data或infer.避免次数太多
    def __init__(self, num_metrics=2):
        assert num_metrics >= 2, 'Evaluation needs at least 2 accumulator metrics'
        super().__init__()
        self.num_metrics = num_metrics
    
    def epoch_judge(self, epoch, num_epochs):
        self.reveal_flag = (epoch+1) % (num_epochs // self.reveal_cnts) == 0 or epoch == 0
        self.eval_flag = (epoch+1) % (num_epochs // self.eval_cnts) == 0 or epoch == 0
        if self.reveal_flag or self.eval_flag:
            self.num_epochs = num_epochs
            self.epoch = epoch
            self.timer = Timer()
            # [0]总train loss, [1]某客观累计量, [3]...
            self.metric = Accumulator(self.num_metrics)
    
    def batch_record(self, net_inputs_batch, loss_inputs_batch, Y_hat, l):
        # Y_hat的shape是(batch_size, num_steps, vocab_size), l的shape是(batch_size,)
        if self.reveal_flag:
            # batch loss(sum of loss of sample), batch num_tokens(sum of Y_valid_lens pf sample)
            self.metric.add(l.sum(), loss_inputs_batch[1].sum())
    
    def epoch_metric_cast(self, fpath, verbose=False):
        '''file path: trainer.log_file_path'''
        if self.reveal_flag:
            time_cost, loss_avg, speed = self.timer.stop(), round(self.metric[0]/self.metric[1],3), round(self.metric[1]/time_cost)
            log = ",".join(['epoch: '+str(self.epoch+1), 'loss(loss/validtoken): '+str(loss_avg), 'speed(tokens/sec): '+str(speed),
                            'remain_time(min): '+str(round(time_cost*(self.num_epochs-self.epoch-1)/60))])
            with open(fpath, 'a+') as f:
                f.write(log+'\n')
            if verbose:
                print(log)