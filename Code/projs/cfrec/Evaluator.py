import os
from ...Compute.EvaluateTools import Timer, Accumulator, epochEvaluator
from ...Compute.VisualizeTools import Animator
import yaml
configs = yaml.load(open('Code/projs/cfrec/configs.yaml', 'rb'), Loader=yaml.FullLoader)
reveal_cnt_in_train, eval_cnt_in_train= configs['reveal_cnt_in_train'], configs['eval_cnt_in_train']
online_log_dir, proj_name = configs['online_log_dir'], configs['proj_name']


class projEpochEvaluator(epochEvaluator):
    reveal_cnts = reveal_cnt_in_train # 披露train情况次数, 从train过程中收集
    eval_cnts = eval_cnt_in_train # 评价当前model, 需要validate data或infer.避免次数太多
    def __init__(self, num_epochs, log_fname, visualizer=None, scalar_names=['loss', ]):
        pass

    def epoch_judge(self, epoch):
        pass
    
    def batch_record(self):
        pass
    
    def evaluate_model(self):
        pass

    def epoch_metric_cast(self):
        '''log file path: ../logs/{}/xxxx.txt'''
        pass