import os
from ...Compute.EvaluateTools import Timer, Accumulator, epochEvaluator
from ...Compute.VisualizeTools import Animator
import yaml
configs = yaml.load(open('src/projs/word2vec/configs.yaml', 'rb'), Loader=yaml.FullLoader)
reveal_cnt_in_train, eval_cnt_in_train= configs['reveal_cnt_in_train'], configs['eval_cnt_in_train']
online_log_dir, proj_name = configs['online_log_dir'], configs['proj_name']


class word2vecEpochEvaluator(epochEvaluator):
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
        self.legends = ['train_loss', ]
        self.visual_flag =  visualizer
        if self.visual_flag:
            self.animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=self.legends)

    def epoch_judge(self, epoch):
        self.reveal_flag = (self.reveal_cnts != 0) and ( (epoch+1) % (self.num_epochs // self.reveal_cnts) == 0 or epoch == 0 )
        self.eval_flag = False # word2vec不评测模型, 只reveal训练情况
        if self.reveal_flag or self.eval_flag:
            self.epoch = epoch
            self.timer = Timer()
            self.reveal_metric = Accumulator(3) if self.reveal_flag else None
            self.eval_metric = None
    
    def batch_record(self, labels, cond_tks, l):
        if self.reveal_flag:
            # batch loss(sum of loss of sample), batch num truth target tokens, batch_size
            self.reveal_metric.add(l.sum(), labels.sum().item(), cond_tks.shape[0])
    
    def evaluate_model(self):
        pass

    def epoch_metric_cast(self):
        '''log file path: ../logs/word2vec/xxxx.txt'''
        loss_avg, reveal_log = None, ''
        if self.reveal_flag:
            time_cost = self.timer.stop()
            loss_avg = round(self.reveal_metric[0]/self.reveal_metric[2], 3)
            speed = round(self.reveal_metric[1]/time_cost)
            reveal_log = ",\t".join(['epoch: '+str(self.epoch+1), 'train_loss(/token): '+str(loss_avg), 'speed(tgt_tk/sec): '+str(speed),
                                     'remain_time(min): '+str(round(time_cost*(self.num_epochs-self.epoch-1)/60))])
            with open(self.log_file, 'a+') as f:
                f.write(reveal_log+'\n')
        if self.visual_flag:
            # 线条的顺序要和legends一一对应. 目前只支持最多4条线
            self.animator.add(self.epoch+1, (loss_avg,))
        if (not self.visual_flag) and (self.reveal_flag or self.eval_flag):
            print(reveal_log + "\n")