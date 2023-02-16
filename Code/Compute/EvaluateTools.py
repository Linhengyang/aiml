import time
import numpy as np
import collections
import math

class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class epochEvaluator(object):
    '''
    train过程中的epoch评价器, 记录每一个epoch的:
        1. train loss(sum of loss among train batch/valid hat nums among train batch) 
        2. train accuracy(sum of correct preds among train batch/nums among train batch)
        3. valid loss(sum of loss among valid batch/valid hat nums among valid batch)(可选)
        4. valid accuracy(sum of correct preds among valid batch/nums among valid batch)
    其中前两个train的指标, 在batch内部记录得到; 后两个valid的指标, 在当前epoch对valid_data作用net得到
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()

    def epoch_judge(self, *args, **kwargs):
        raise NotImplementedError

    def batch_record(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate_model(self, *args, **kwargs):
        raise NotImplementedError

    def epoch_metric_cast(self, *args, **kwargs):
        raise NotImplementedError

def bleu(pred_seq, label_seq, k):
    """计算BLEU. 可以处理pred_seq为空字符串的情况
    inputs:
        1. pred_seq和label_seq: 输入前需要lower/替换非正常空格为单空格/文字和,.?!之间需要有单空格
        2. k: 用于匹配的n-gram的最大长度, k <= min(len(pred_seq), len(label_seq))
    """
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    if k > min(len_pred, len_label): # 如果k超出了pred_seq和label_seq的其中之一的长度, 限定k
        k = min(len_pred, len_label)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

if __name__ == "__main__":
    print(bleu("he\'s calm .", 'il est calme .', 5))