import random
import numpy as np

class RandomGenerator:
    '''
    形式上: 每次draw, 都是以sampling_weights = {p1, p2, ..., pN}的概率分布, 从{1, 2, ..., N}中选取其一
    实际上: 以sampling_weights = {p1, p2, ..., pN}的概率分布, 从{1, 2, ..., N}中, 一次性sample cache_size个,
    然后在调用.draw()的时候逐一返回. 当sample好的用完了之后, 在下一次draw时会自动重新sample cache_size个
    '''
    def __init__(self, sampling_weights, cache_size=10000):
        self.population = list(range(1, len(sampling_weights)+1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0
        self.cache_size = cache_size
    
    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = random.choices(self.population, self.sampling_weights, self.cache_size)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]