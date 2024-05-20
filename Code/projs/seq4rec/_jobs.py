from .Network import seq4recEncoder
from .Trainer import seq4recTrainer
from .Dataset import seq4recDataset
import torch
import torch.nn as nn


def train_job():
    # trainset = seq4recDataset(path="../data/seq4rec/train_data.csv")
    trainset = seq4recDataset(path="../data/seq4rec/train_data.db", tbl_name='seq4rec_train')
    num_hiddens, dropout = 128, 0.1
    net = seq4recEncoder(num_hiddens, dropout)
    loss = nn.CrossEntropyLoss()
    # train_iter = torch.utils.data.DataLoader(trainset, 10, True)
    # for X, y in train_iter:
    #     break

    # y_hat = net(X)
    # l = loss(y_hat, y)
    # print('y_hat:', y_hat, '\n', 'y:', y, '\n', 'loss:', l)

    trainer = seq4recTrainer(net, loss, 2, 1000)
    trainer.set_data_iter(trainset)
    trainer.set_optimizer(0.0005 )## set the optimizer
    trainer.resolve_net(True)
    trainer.log_topology()
    # fit
    trainer.fit()
    # save
    trainer.save_model()





if __name__ == "__main__":
    pass
    # train_iter = torch.utils.data.DataLoader(trainset, 10, True)
    # for X, y in train_iter:
    #     break

    # print("X:", X, '\n', 'y:', y)