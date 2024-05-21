import torch.utils
from .Network import seq4recEncoder
from .Trainer import seq4recTrainer
from .Dataset import seq4recDataset
from .Predictor import seq4recPredictor
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


def train_job():
    # trainset = seq4recDataset(path="../data/seq4rec/train_data.csv")
    trainset = seq4recDataset(path="../data/seq4rec/train_data.db", tbl_name='seq4rec_train', mode='train')
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

def infer_job():
    inferset = seq4recDataset(path="../data/seq4rec/train_data.db", tbl_name='seq4rec_train', mode='infer')

    # import model
    num_hiddens, dropout = 128, 0.1
    net = seq4recEncoder(num_hiddens, dropout)
    net.load_state_dict(torch.load("../model/seq4rec/seq4rec.params"))

    # infer
    net.eval()
    result = torch.zeros(0,2)
    with torch.no_grad():
        for X, user_ids in torch.utils.data.DataLoader(inferset, 10000, False):

            cur_scores = nn.functional.sigmoid( net(X))[:, 1]
            cur_result = torch.stack([user_ids, cur_scores], dim=1)

            result = torch.cat([result, cur_result], dim=0)

    result_df = pd.DataFrame( result.numpy() ) #convert to a dataframe
    result_df.to_csv("../logs/seq4rec/output.csv", index=False) #save to file








if __name__ == "__main__":
    pass
    # train_iter = torch.utils.data.DataLoader(trainset, 10, True)
    # for X, y in train_iter:
    #     break

    # print("X:", X, '\n', 'y:', y)