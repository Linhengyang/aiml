import os
from ...Compute.EvaluateTools import Timer, Accumulator, epochEvaluator, metric_summary
from ...Compute.VisualizeTools import Animator
import yaml


configs = yaml.load(open('src/projs/gpt/configs.yaml', 'rb'), Loader=yaml.FullLoader)
reveal_cnt_in_train, eval_cnt_in_train= configs['reveal_cnt_in_train'], configs['eval_cnt_in_train']


class projEpochEvaluator(epochEvaluator):
    # class 变量
    reveal_cnts = reveal_cnt_in_train # 披露train情况次数, 从train过程中收集
    eval_cnts = eval_cnt_in_train # 评价当前model, 需要validate data或infer.避免次数太多

    def __init__(self, num_epochs, logfile_path, scalar_names=['loss', ], dim_accum=2, 
                 visualizer=None, verbose=False):
        
        assert num_epochs >= max(self.reveal_cnts, self.eval_cnts), \
            f'num_epochs must be larger than reveal counts & eval counts'
        
        self.num_epochs = num_epochs
        self.reveal_accumulator, self.eval_accumulator = Accumulator(dim_accum), Accumulator(dim_accum)

        # 设定 日志文件 地址, 并打印 train begin
        self.log_file = logfile_path
        with open(self.log_file, 'w') as f:
            print('train begin', file=f)

        # 记录 标签
        self.legends = []
        for name in scalar_names:
            if self.reveal_cnts != 0:
                self.legends.append('train_'+name)
            if self.eval_cnts != 0:
                self.legends.append('valid_'+name)
        
        # 图像显示器
        self.visual_flag = True if visualizer else False
        if self.visual_flag:
            self.animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=self.legends)

        # 是否需要在控制台 打印训练日志
        self.verbose_flag = verbose


    # 确定当前 epcoh 要不要作 reveal train situation 或 evaluate current validation situation
    def epoch_judge(self, epoch):
        '''
        训练起始(epoch=0)时要 reveal train situation, 也要 evaluate validation situation
        在训练过程中, 要满足 reveal 总次数 接近设定好的 reveal_cnts, eval 总次数 接近设定好的 eval_cnts
        '''
        self.reveal_flag = (self.reveal_cnts != 0) and ( (epoch+1) % (self.num_epochs // self.reveal_cnts) == 0 or epoch == 0 )
        self.eval_flag = (self.eval_cnts != 0) and ( (epoch+1) % (self.num_epochs // self.eval_cnts) == 0 or epoch == 0 )
        self.epoch = epoch

        # 若当前 epoch 需要 reveal train, 开始计时, reveal累加器二位(train loss, num_tokens)
        if self.reveal_flag:
            self.timer = Timer()

    
    def batch_record(self, *args, **kwargs):
        pass
    

    def evaluate_model(self, *args, **kwargs):
        pass


    def epoch_metric_cast(self):
        '''log file path: ../logs/projName/xxxx.txt'''
        pass