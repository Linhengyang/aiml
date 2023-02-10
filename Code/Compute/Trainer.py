
class Trainer(object):
    '''
    components: 
        net, loss, optimizer(None), train_data_iter, valid_data_iter(None), grad_clip_val(None), num_epochs, device(None)
    '''
    def __init__(self):
        super().__init__()
    
    def param_initializer(self, *args, **kwargs):
        '''对net的parameters作初始化'''
        raise NotImplementedError
    
    def device_manager(self, *args, **kwargs):
        '''对train过程的设备device作管理'''
        raise NotImplementedError
    
    def param_optimizer(self, *args, **kwargs):
        '''设定train过程的优化器'''
        raise NotImplementedError
    
    def grad_clipper(self, *args, **kwargs):
        '''train过程的参数裁剪'''
        raise NotImplementedError
    
    def evaluator(self, *args, **kwargs):
        '''train过程中的评价器'''
        raise NotImplementedError
    
    def visualizer(self, *args, **kwargs):
        '''可视化train过程'''
        raise NotImplementedError
    
    def fit(self, *args, **kwargs):
        '''run整个train过程'''
        raise NotImplementedError