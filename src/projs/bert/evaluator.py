from src.core.evaluation.evaluate import Timer, Accumulator, metric_summary
from src.core.interface.infra_easy import epochEvaluator
from src.utils.visualize import Animator
import yaml
import typing as t



configs = yaml.load(open('src/projs/bert/configs.yaml', 'rb'), Loader=yaml.FullLoader)


class bertEpochEvaluator(epochEvaluator):

    reveal_cnts = configs['reveal_cnt_in_train'] # 披露train情况的次数, 从train过程中收集
    eval_cnts = configs['eval_cnt_in_train']     # 评价当前model, 需要infer on validation data. 避免次数太多

    def __init__(self, num_epochs, logfile_path, scalar_names=['mlm_loss', 'nsp_loss', ], num_dims_for_accum=4,
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
        
        # train_legend1, valid_legend1, train_legend2, valid_legend2,...

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
    def record_batch(self, net_inputs_batch, loss_inputs_batch, mlm_l, nsp_l):
        # net_inputs_batch        
        # tokens: (batch_size, seq_len)int64 ot token ID. 已包含<cls>和<sep>
        # valid_lens: (batch_size,)
        # segments: (batch_size, seq_len)01 indicating seq1 & seq2 | None, None 代表当前 batch 不需要进入 NSP task
        # mask_positions: (batch_size, num_masktks) | None, None 代表当前 batch 不需要进入 MLM task

        # loss_inputs_batch
        # mlm_valid_lens (batch_size,)
        # mlm_label (batch_size, num_masktks)
        # nsp_label (batch_size,)
        if self.reveal_flag:
            # 记录 batch mlm loss, batch nsp loss, batch valid masked tokens 总数量, batch size
            self.reveal_accumulator.add(mlm_l.sum(), nsp_l.sum(),
                                        loss_inputs_batch[0].sum(), loss_inputs_batch[2].dim())
    
    # @evaluation: record values for scalars
    def evaluate_model(self, net, loss, valid_iter, FP_step:t.Callable, num_batches=None):
        if self.eval_flag and valid_iter: # 如果此次 epoch 确定要 evaluate network, 且输入了 valid_iter
            net.eval() # bert 可以把 net 设为 eval 模式
            for i, (net_inputs_batch, loss_inputs_batch) in enumerate(valid_iter):
                if num_batches and i >= num_batches:
                    break
                # net_inputs_batch
                # tokens: (batch_size, seq_len)int64 ot token ID. 已包含<cls>和<sep>
                # valid_lens: (batch_size,)
                # segments: (batch_size, seq_len)01 indicating seq1 & seq2 | None, None 代表当前 batch 不需要进入 NSP task
                # mask_positions: (batch_size, num_masktks) | None, None 代表当前 batch 不需要进入 MLM task

                # loss_inputs_batch
                # mlm_valid_lens (batch_size,)
                # mlm_label (batch_size, num_masktks)
                # nsp_label (batch_size,)
                mlm_l, nsp_l, _ = FP_step(net, loss, net_inputs_batch, loss_inputs_batch)
                # 记录 batch mlm loss, batch nsp loss, batch valid masked tokens 总数量, batch size
                self.eval_accumulator.add(mlm_l.sum(), nsp_l.sum(),
                                          loss_inputs_batch[0].sum(), loss_inputs_batch[2].dim())
    
    def cast_metric(self):
        # 若当前 epoch 需要 reveal train, 停止计时, reveal累加器4位( mlm loss, nsp loss, valid masked tokens 总数量, size)
        if self.reveal_flag and self.reveal_accumulator:
            time_cost = self.timer.stop()
            mlm_loss_train = self.reveal_accumulator[0] / self.reveal_accumulator[3]
            nsp_loss_train = self.reveal_accumulator[1] / self.reveal_accumulator[3]
            loss_train = mlm_loss_train + nsp_loss_train
            
            speed = self.reveal_accumulator[2] / time_cost
            est_remain_time = time_cost*(self.num_epochs-self.epoch-1)/60

            reveal_log = metric_summary(
                values = [self.epoch+1, mlm_loss_train, nsp_loss_train, loss_train, speed, est_remain_time],
                metric_names = ["epoch", "train_mlm_loss", "train_nsp_loss", "train_loss", "speed", "remain_time"],
                unit_names = ["", "/token", "/sample", "/sample", "tokens/sec", "min"],
                round_ndigits = [None, 3, 3, 3, 0, 1]
                )
            with open(self.log_file, 'a+') as f:
                f.write(reveal_log+'\n')
            if self.verbose_flag:
                print(reveal_log)
        
        # 若当前 epoch 需要 evaluate model, eval累加器4位( mlm loss, nsp loss, valid masked tokens 总数量, size)
        if self.eval_flag and self.eval_accumulator:
            mlm_loss_eval = self.eval_accumulator[0] / self.eval_accumulator[3]
            nsp_loss_eval = self.eval_accumulator[1] / self.eval_accumulator[3]
            loss_eval = mlm_loss_eval + nsp_loss_eval

            eval_log = metric_summary(
                values = [self.epoch+1, mlm_loss_eval, nsp_loss_eval, loss_eval],
                metric_names = ["epoch", "eval_mlm_loss", "eval_nsp_loss", "eval_loss"],
                unit_names = ["", "/token", "/sample", "/sample",],
                round_ndigits = [None, 3, 3, 3]
            )
            with open(self.log_file, 'a+') as f:
                f.write(eval_log+'\n')
            if self.verbose_flag:
                print(eval_log)

        # 若设定了 visualizer
        if self.visual_flag:
            mlm_loss_train = mlm_loss_train if self.reveal_flag else None
            mlm_loss_eval = mlm_loss_eval if self.eval_flag else None
            nsp_loss_train = nsp_loss_train if self.reveal_flag else None
            nsp_loss_eval = nsp_loss_eval if self.eval_flag else None
            # 线条的顺序要和legends一一对应. 目前只支持最多4条线
            # self.legends: (train_mlm_loss, valid_mlm_loss, train_nsp_loss, valid_nsp_loss)
            self.animator.add(self.epoch+1, (mlm_loss_train, mlm_loss_eval, nsp_loss_train, nsp_loss_eval))

        # 本epoch cast 成功之后, 清空 两个accumulator
        self.reveal_accumulator.reset()
        self.eval_accumulator.reset()