
class easyTrainer(object):
    '''
    components: 
        net, loss, num_epochs, batch_size
    单机单卡训练任务的trainer接口
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()

    def log_topology(self, *args, **kwargs):
        '''log the topology of the network to topos directory'''
        raise NotImplementedError

    def set_device(self, *args, **kwargs):
        '''指定trainer的设备'''
        raise NotImplementedError

    def set_data_iter(self, *args, **kwargs):
        '''
        设定trainer的data_iters, 包括:
            train_data_iter(必须), valid_data_iter(可选), test_data_iter(可选, resolve网络时用)
        '''
        raise NotImplementedError

    def resolve_net(self, *args, **kwargs):
        '''
        用test_data_iter的first batch对net作一次forward计算, 使得所有lazyLayer被确定(resolve).
        随后检查整个前向过程和loss计算(check).
        resolve且check无误 --> True
        resolve或check有问题 --> raise AssertionError
        不resolve或check直接训练 --> False
        '''
        raise NotImplementedError
    
    def init_params(self, *args, **kwargs):
        '''对net的parameters作初始化'''
        raise NotImplementedError
    
    def set_optimizer(self, *args, **kwargs):
        '''设定train过程的优化器'''
        raise NotImplementedError
    
    def set_grad_clipping(self, *args, **kwargs):
        '''train过程的参数裁剪'''
        raise NotImplementedError
    
    def set_epoch_eval(self, *args, **kwargs):
        '''train过程中的评价器'''
        raise NotImplementedError
    
    def save_model(self, *args, **kwargs):
        '''save the model to model directory'''
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError
    


class easyPredictor(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def set_device(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError
    
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError
    
    @property
    def pred_scores(self):
        pass



class epochEvaluator(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def judge_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def record_batch(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate_model(self, *args, **kwargs):
        raise NotImplementedError

    def cast_metric(self, *args, **kwargs):
        raise NotImplementedError
    