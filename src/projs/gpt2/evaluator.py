# evaluator.py for gpt2
from ...core.base.compute.evaluate_tools import Timer, Accumulator, epochEvaluator, metric_summary
import torch
import yaml

configs = yaml.load(open('src/projs/gpt2/configs.yaml', 'rb'), Loader=yaml.FullLoader)


class gpt2EpochEvaluator(epochEvaluator):

    reveal_cnts = configs['reveal_cnt_in_train'] # 披露train情况次数, 从train过程中收集

    def __init__(self, num_epochs, logfile_path):
        assert num_epochs >= self.reveal_cnts
        super().__init__()
        self.num_epochs = num_epochs
        self.reveal_accumulator = Accumulator(2)

        self.log_file = logfile_path
        with open(self.log_file, 'w') as f:
            print('train begin', file=f)

    def judge_epoch(self, epoch):
        self.reveal_flag = (self.reveal_cnts != 0) and ( (epoch+1) % (self.num_epochs // self.reveal_cnts) == 0 or epoch == 0 )
        self.epoch = epoch

        if self.reveal_flag:
            self.timer = Timer()
    
    def record_batch(self, l):
        if self.reveal_flag:
            with torch.no_grad():
                # 记录 batch loss 和 token 数量
                self.reveal_accumulator.add(l.sum(), l.numel())

    def cast_metric(self):
        # 若当前 epoch 需要 reveal train, 停止计时, reveal 累加器二位(train loss, num_tokens)
        if self.reveal_flag:
            time_cost = self.timer.stop()
            loss_avg = self.reveal_accumulator[0] / self.reveal_accumulator[1]
            speed = self.reveal_accumulator[1] / time_cost
            est_remain_time = time_cost*(self.num_epochs-self.epoch-1)/60

            reveal_log = metric_summary(
                values = [self.epoch+1, loss_avg, speed, est_remain_time],
                metric_names = ["epoch", "train_loss", "speed", "remain_time"],
                unit_names = ["", "/token", "tokens/sec", "min"],
                round_ndigits = [None, 3, 0, 0]
                )
            
            with open(self.log_file, 'a+') as f:
                f.write(reveal_log+'\n')
            
            self.reveal_accumulator.reset()