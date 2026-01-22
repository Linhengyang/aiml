import os
from ...core.evaluation.evaluate import Timer, Accumulator, accuracy, metric_summary
from ...core.interface.simple import epochEvaluator
from ...core.base.tool.visualize import Animator
import yaml
configs = yaml.load(open('src/projs/vit/configs.yaml', 'rb'), Loader=yaml.FullLoader)
reveal_cnt_in_train, eval_cnt_in_train= configs['reveal_cnt_in_train'], configs['eval_cnt_in_train']



class vitEpochEvaluator(epochEvaluator):

    # class 变量
    reveal_cnts = reveal_cnt_in_train # 披露train情况次数, 从train过程中收集
    eval_cnts = eval_cnt_in_train # 评价当前model, 需要validate data或infer.避免次数太多

    def __init__(self, num_epochs, logfile_path, scalar_names=['loss', 'accuracy'], num_dims_for_accum=3,
                 visualizer=None, verbose=False):
        
        assert num_epochs >= max(self.reveal_cnts, self.eval_cnts), \
            f'num_epochs must be larger than reveal counts & eval counts'
        
        super().__init__()
        
        self.num_epochs = num_epochs
        self.reveal_accumulator, self.eval_accumulator = Accumulator(num_dims_for_accum), Accumulator(num_dims_for_accum)

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
    def judge_epoch(self, epoch):
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
    
    # @train: record values for scalars
    def record_batch(self, X, y, Y_hat, l):
        # X: (batch_size, num_channels, height, width)
        # y: (batch_size,)
        # Y_hat: (batch_size, num_classes)
        # l: (batch_size, )
        if self.reveal_flag:
            # batch loss(sum of loss of sample), batch num correct preds, batch num_imgs(sum of y pf sample)
            self.reveal_accumulator.add(l.sum(), accuracy(Y_hat, y), y.numel())
    

    # @evaluation: record values for scalars
    def evaluate_model(self, net, loss, valid_iter, num_batches=None):
        if self.eval_flag and valid_iter: # 如果此次 epoch 确定要 evaluate network, 且输入了 valid_iter
            net.eval()
            for i, (X, y) in enumerate(valid_iter): # 如果 设定了 evaluate 的 batch 数量，那么在达到时，退出 eval_metric 的累积
                if num_batches and i >= num_batches:
                    break
                Y_hat = net(X)
                l = loss(Y_hat, y)
                # batch loss(sum of loss of sample), batch num correct preds, batch num_imgs(sum of y pf sample)
                self.eval_accumulator.add(l.sum(), accuracy(Y_hat, y), y.numel())


    def cast_metric(self):
        
        loss_avg, acc_avg, eval_loss_avg, eval_acc_avg = None, None, None, None

        # 若当前 epoch 需要 reveal train, 停止计时, reveal累加器二位(train loss, num_tokens)
        if self.reveal_flag:
            # reveal_accumulator: loss, num correct preds, num_imgs

            time_cost = self.timer.stop()
            loss_avg = self.reveal_accumulator[0] / self.reveal_accumulator[2]
            acc_avg = self.reveal_accumulator[1] / self.reveal_accumulator[2] * 100
            speed = self.reveal_accumulator[2] / time_cost
            est_remain_time = time_cost*(self.num_epochs-self.epoch-1)/60

            reveal_log = metric_summary(
                values = [self.epoch+1, loss_avg, acc_avg, speed, est_remain_time],
                metric_names = ["epoch", "train_loss", "train_accuracy", "speed", "remain_time"],
                unit_names = ["", "/img", "%", "imgs/sec", "min"],
                round_ndigits = [None, 3, 1, 0, 0]
                )
            
            with open(self.log_file, 'a+') as f:
                f.write(reveal_log+'\n')
            
            if self.verbose_flag:
                print(reveal_log)
        
        
        # 若当前 epoch 需要 evaluate model, reveal累加器二位(validation loss, num_tokens)
        if self.eval_flag:
            # eval_accumulator: loss, num correct preds, num_imgs

            eval_loss_avg = self.eval_accumulator[0] / self.eval_accumulator[2]
            eval_acc_avg = self.eval_accumulator[1] / self.eval_accumulator[2] * 100

            eval_log = metric_summary(
                values = [self.epoch+1, eval_loss_avg, eval_acc_avg],
                metric_names = ["epoch", "eval_loss", "eval_accuracy"],
                unit_names = ["", "/img", "%"],
                round_ndigits = [None, 3, 1]
            )

            with open(self.log_file, 'a+') as f:
                f.write(eval_log+'\n')

            if self.verbose_flag:
                print(eval_log)

        # 若设定了 visualizer
        if self.visual_flag:
            loss_avg = loss_avg if self.reveal_flag else None
            eval_loss_avg = eval_loss_avg if self.eval_flag else None
            acc_avg = acc_avg if self.reveal_flag else None
            eval_acc_avg = eval_acc_avg if self.eval_flag else None

            # 线条的顺序要和legends一一对应. 目前只支持最多4条线
            # self.legends: (train_loss, valid_loss, train_accuracy, valid_accuracy)
            self.animator.add(self.epoch+1, (loss_avg, eval_loss_avg, acc_avg, eval_acc_avg))

        # 本epoch cast 成功之后, 清空 两个accumulator
        self.reveal_accumulator.reset()
        self.eval_accumulator.reset()