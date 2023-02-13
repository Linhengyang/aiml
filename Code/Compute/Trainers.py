
class easyTrainer(object):
    '''
    components: 
        net, loss, optimizer(None), train_data_iter, valid_data_iter(None), 
        grad_clip_val(None), num_epochs, device(None)
    单机单卡
    '''
    def __init__(self):
        super().__init__()
    
    def log_topology(self, *args, **kwargs):
        '''log the topology of the network to topos directory'''
        raise NotImplementedError
    
    def set_data_iter(self, *args, **kwargs):
        raise NotImplementedError

    def resolve_net(self, *args, **kwargs):
        '''
        用train_data_iter的first batch对net作一次forward计算, 使得所有lazyLayer被确定,
        然后再初始化参数
        '''
        raise NotImplementedError
    
    def init_params(self, *args, **kwargs):
        '''对net的parameters作初始化'''
        raise NotImplementedError
    
    def set_device(self, *args, **kwargs):
        '''对train过程的设备device作管理
        1. net移动到device上
        2. train data batch移动到device上
        '''
        raise NotImplementedError
    
    def set_optimizer(self, *args, **kwargs):
        '''设定train过程的优化器'''
        raise NotImplementedError
    
    def set_grad_clipping(self, *args, **kwargs):
        '''train过程的参数裁剪'''
        raise NotImplementedError
    
    def evaluator(self, *args, **kwargs):
        '''train过程中的评价器'''
        raise NotImplementedError
    
    def visualizer(self, *args, **kwargs):
        '''可视化train过程'''
        raise NotImplementedError
    
    def save_model(self, *args, **kwargs):
        '''save the model to model directory'''
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        '''run整个train过程:
        device, topo_logger, net_resolver, topo_logger, param_initializer, train
        '''
        raise NotImplementedError