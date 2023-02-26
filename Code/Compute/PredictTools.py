class easyPredictor(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def set_device(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError
    
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError
    
    @property
    def pred_scores(self):
        pass