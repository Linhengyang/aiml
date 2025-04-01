import time
import numpy as np
import collections
import math
import torch
from torch import Tensor
import typing as t

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


def metric_summary(
        values, 
        metric_names:t.List[str], 
        unit_names:t.List[str], 
        round_ndigits:t.List[int|None], 
        separator='\t'
        ) -> str:

    names = [metric + "(" + unit + "): " for metric, unit in zip(metric_names, unit_names)]

    vals = [round(value, round_ndigit) if round_ndigit else value for value, round_ndigit in zip(values, round_ndigits)]

    name_value_pairs = [name+str(val) for name, val in zip(names, vals)]

    return separator.join(name_value_pairs)





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






def accuracy(y_hat: Tensor, y: Tensor):
    """
    计算预测正确的数量, 类似nn.CrossEntropyLoss
    y_hat: (batch_size, num_classes, positions(optional)), elements是logit或softmax后的Cond Prob
    y: (batch_size, positions(optional)), elements是label(非one-hot), dtype是torch.int64
    输出整个batch中预测正确的个体数量(不是样本数量), 相当于CELoss reduction='sum'
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())






def binary_accuracy(y_hat: Tensor, y: Tensor, threshold=0.5):
    """
    计算二值预测中正确的数量, 类似nn.BCELoss(Binary Cross Entropy Loss)
    y_hat: (batch_size, positions(optional)), elements是logit或softmax后的Cond Prob
    y: (batch_size, positions(optional)), elements是0或1, dtype是torch.int64
    threshold: 预测为1的阈值
    输出整个batch中预测正确的个体数量(不是样本数量), 相当于BCELoss reduction='sum'
    """
    cmp = (y_hat > threshold).type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 二分类问题的四种预测结果
# True Positive: 正确地判别成1, 即Truth为1的元素, 预测为1
# False Negative: 错误地判别成0, 即Truth为1的元素, 预测为0
# True Negative: 正确地判别成0, 即Truth为0的元素, 预测为0
# False Positive: 错误地判别成1, 即Truth为0的元素, 预测为1





def confuse_mat(y_hat: Tensor, y: Tensor, threshold=0.5):
    '''
    y_hat & y shape: (*), same shape
    y should be 0-1 tensor
    '''
    truth_p_mask = torch.sign(y).type(torch.bool)
    TP = binary_accuracy(y_hat[truth_p_mask], y[truth_p_mask], threshold)
    FN = binary_accuracy(y_hat[truth_p_mask], 1-y[truth_p_mask], threshold)
    TN = binary_accuracy(y_hat[~truth_p_mask], y[~truth_p_mask], threshold)
    FP = binary_accuracy(y_hat[~truth_p_mask], 1-y[~truth_p_mask], threshold)
    assert TP + FN == y[truth_p_mask].numel()
    assert TN + FP == y[~truth_p_mask].numel()
    return {'TP':TP, 'FN':FN, 'TN':TN, 'FP':FP}





def binary_classify_eval_rates(y_hat: Tensor, y: Tensor, threshold=0.5):
    '''
    y_hat & y shape: (*), same shape
    y should be 0-1 tensor
    '''
    confuse_mat = confuse_mat(y_hat, y, threshold)
    TP, FN, TN, FP = confuse_mat['TP'], confuse_mat['FN'], confuse_mat['TN'], confuse_mat['FP']
    acc_rt, precision, recall = (TP+TN)/(TP+FN+TN+FP), TP/(TP+FP), TP/(TP+FN)
    FPR = FP/(FP+TN)
    return {'acc_rt':acc_rt, 'precision':precision, 'recall':recall, 'TPR':recall, 'FPR':FPR}






# ROC曲线: 给定一个二分类器和阈值, 作用在样本V中, 关注预测为Positive的样本, 可得到TPR和FPR
# TPR = TP/(TP+FN), 所有真实为P的样本中, 预测为P的比例, 就是recall
# FPR = FP/(FP+TN), 所有真实为N的样本中, 预测为P的比例
# 以FPR为横坐标, TPR为纵坐标, plot点. 显然TPR接近1越好, FPR越接近0越好, 所以点在左上角(0, 1)最好
# 针对一个二分类器, 滑动阈值从0到1.
# 当阈值为0时, 所有样本预测为1, 此时FN=TN=0, TPR=FPR=1, 点在(1, 1)
# 当阈值为1时, 所有样本预测为0, 此时TP=FP=0, TPR=FPR=0, 点在(0, 0)
# 当阈值从1到0时, 由于标准放低, 更多样本被预测为1, FN和TN都会减小或持平(二者之和会减小), 所以TPR和FPR都会变大或持平
def roc_curve(net, sample):
    raise NotImplementedError







# AUC指标: ROC曲线下面的面积, 代表「随机给定一对正负样本, 存在一个阈值可用在该分类器上将二者分开」的概率, 即「随机给定一对正负样本, 二者的预测序正确」的概率
def auc(preds :Tensor, labels: Tensor) -> tuple:
    '''
    preds & labels shape: (batch_size, )
    labels are 0-1 tensor
    '''
    pos_preds = preds[ torch.sign(labels).type(torch.bool) ] # 正样本的preds, shape(num_pos, )
    neg_preds = preds[ ~torch.sign(labels).type(torch.bool) ] # 负样本的preds, shape(num_neg, )
    cmp_cnt = pos_preds.shape[0] * neg_preds.shape[0]
    if cmp_cnt > 0:
        correct_cnt = ( pos_preds.unsqueeze(0) > neg_preds.unsqueeze(1) ).sum().item()
        auc = correct_cnt/cmp_cnt
    else: # 如果可比正负对数为0, 则跳过本次比较
        correct_cnt = 0
        auc = 0
    return (auc, correct_cnt, cmp_cnt)