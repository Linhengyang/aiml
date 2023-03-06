import random
import numpy as np

class RandomGenerator:
    '''
    形式上: 每次draw, 都是以
        sampling_weights = [p1, p2, ..., pN]
    的概率分布, 从population = [v1, v2, ..., vN]中选取其一
    实际上: 以
        sampling_weights = [p1, p2, ..., pN]
    的概率分布, 从population = [v1, v2, ..., vN]中一次性取样 cache_size个,
    然后在调用.draw()的时候逐一返回. 当sample好的用完了之后, 在下一次draw时会自动重新取样 cache_size个
    '''
    def __init__(self, population: list, sampling_weights: list, cache_size=10000):
        assert len(sampling_weights) == len(population), 'population & sampling_weights length mismatch'
        self.population = population
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


class NegativeSampling:
    def __init__(self, population: list, sampling_weights: list, inflation_size: int, cache_size: int = 10000):
        self.population = population
        self.sampling_weights = sampling_weights
        self.inflation_size = inflation_size
        self.cache_size = cache_size
    
    @staticmethod
    def _sample_for_record(label, generator: RandomGenerator, inflation_size: int):
        '''
        input:
            label: 单行的正label, 可以是多个tokens/items组成的list
            generator: 在population中以sampling_weights概率分布采样
            inflation_size: 负样本扩增倍率
        '''
        if not isinstance(label, list):
            label = [label, ]
        negatives_record = []
        while len(negatives_record) < len(label) * inflation_size:
            candidate = generator.draw()
            if not candidate in label:
                negatives_record.append(candidate)
        return negatives_record

    def sample(self, labels: list):
        '''
        input:
            labels: sample_size行的正labels, len(labels) = sample_size, 每行的正label可以是多个tokens/items组成的list
        return:
            negatives: sample_size行的负labels, len(negatives) = sample_size
        '''
        negatives = []
        generator = RandomGenerator(self.population, self.sampling_weights, self.cache_size)
        for label in labels:
            negatives.append(self._sample_for_record(label, generator, self.inflation_size))
        return negatives

