def accuracy(y_hat, y):  #@save
    """
    计算预测正确的数量, 类似CrossEntropyLoss
    y_hat: (batch_size, num_classes, positions(optional)), elements是logit或softmax后的Cond Prob
    y: (batch_size, positions(optional)), elements是label(非one-hot), dtype是torch.int64
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())