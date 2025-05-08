from ...Compute.EvaluateTools import Timer, Accumulator, epochEvaluator, metric_summary
from ...Compute.VisualizeTools import Animator
import yaml


configs = yaml.load(open('Code/projs/transformer/configs.yaml', 'rb'), Loader=yaml.FullLoader)
reveal_cnt_in_train, eval_cnt_in_train= configs['reveal_cnt_in_train'], configs['eval_cnt_in_train']


class transformerEpochEvaluator(epochEvaluator):

    # class 变量
    reveal_cnts = reveal_cnt_in_train # 披露train情况次数, 从train过程中收集
    eval_cnts = eval_cnt_in_train # 评价当前model, 需要validate data或infer.避免次数太多

    def __init__(self, num_epochs, logfile_path, scalar_names=['loss', ], num_dims_for_accum=2,
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
    

    # @train: record values for scalars
    def batch_record(self, net_inputs_batch, loss_inputs_batch, Y_hat, l):
        # net_inputs_batch = (X, Y_frontshift1, X_valid_lens)
        # shapes: (batch_size, num_steps), (batch_size, num_steps), (batch_size,)
        # loss_inputs_batch = (Y, Y_valid_lens)
        # shapes: (batch_size, num_steps), (batch_size,)
        if self.reveal_flag:
            # 记录 batch loss, 和 target batch valid token 数量
            self.reveal_accumulator.add(l.sum(), loss_inputs_batch[1].sum())
    

    # @evaluation: record values for scalars
    def evaluate_model(self, net, loss, valid_iter, num_batches=None):
        if self.eval_flag and valid_iter: # 如果此次 epoch 确定要 evaluate network, 且输入了 valid_iter
            # 始终保持 net 在 train mode. 因为 net 在 eval mode 会触发自回归调用 KV_Cache
            for i, (net_inputs_batch, loss_inputs_batch) in enumerate(valid_iter):
                # net_inputs_batch = (X, Y_frontshift1, X_valid_lens)
                # shapes: (batch_size, num_steps), (batch_size, num_steps), (batch_size,)
                # loss_inputs_batch = (Y, Y_valid_lens)
                # shapes: (batch_size, num_steps), (batch_size,)

                if num_batches and i >= num_batches: # 如果 设定了 evaluate 的 batch 数量，那么在达到时，退出 eval_metric 的累积
                    break
                Y_hat, _ = net(*net_inputs_batch)

                l = loss(Y_hat, *loss_inputs_batch)

                # 记录 batch loss, 和 target batch valid token 数量
                self.eval_accumulator.add(l.sum(), loss_inputs_batch[1].sum())



    def epoch_metric_cast(self):

        loss_avg, eval_loss_avg = None, None

        # 若当前 epoch 需要 reveal train, 停止计时, reveal累加器二位(train loss, num_tokens)
        if self.reveal_flag:
            # reveal_accumulator: loss, num_valid_tokens

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
            
            if self.verbose_flag:
                print(reveal_log)
        
        
        # 若当前 epoch 需要 evaluate model, reveal累加器二位(validation loss, num_tokens)
        if self.eval_flag:
            # eval_accumulator: loss, num_valid_tokens

            eval_loss_avg = self.eval_accumulator[0] / self.eval_accumulator[1]

            eval_log = metric_summary(
                values = [self.epoch+1, eval_loss_avg],
                metric_names = ["epoch", "eval_loss"],
                unit_names = ["", "/token"],
                round_ndigits = [None, 3]
            )

            with open(self.log_file, 'a+') as f:
                f.write(eval_log+'\n')

            if self.verbose_flag:
                print(eval_log)

        # 若设定了 visualizer
        if self.visual_flag:
            loss_avg = loss_avg if self.reveal_flag else None
            eval_loss_avg = eval_loss_avg if self.eval_flag else None

            # 线条的顺序要和legends一一对应. 目前只支持最多4条线
            # self.legends: (train_loss, valid_loss)
            self.animator.add(self.epoch+1, (loss_avg, eval_loss_avg))

        # 本epoch cast 成功之后, 清空 两个accumulator
        self.reveal_accumulator.reset()
        self.eval_accumulator.reset()