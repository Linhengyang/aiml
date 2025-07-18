
import abc


class MemorySwitch(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    @abc.abstractmethod
    def allocate_memory(self, *args, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def release_memory(self, *args, **kwargs):
        raise NotImplementedError