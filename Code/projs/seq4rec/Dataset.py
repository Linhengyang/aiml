import torch
import pandas as pd
import numpy as np
from ...Data.dbconnect import DatabaseConnection
import typing as t

def row_parse(array:np.ndarray):
    cat_row = "|".join( array.tolist() )
    cat_row = cat_row.replace("\n", "")
    return np.array( cat_row.split("|") )


class seq4recDataset(torch.utils.data.Dataset):
    def __init__(self, path, tbl_name='csv', mode='train', *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tbl_name == 'csv':
            data = pd.read_csv(path, header=0, dtype=str)
        elif tbl_name == 'seq4rec_train':
            db_info = {
                'path':path
                }
            db = DatabaseConnection(db_info)

            sql_query =\
            '''
            SELECT 
                user_id,
                net_investment_amount,
                trading_amount,
                num_buy_stocks,
                num_sell_stocks,
                trading_frequency,
                if_margin_trading
            FROM {tbl_name}
            '''.format(tbl_name=tbl_name)
            data = db.GetSQL(sql_query)
            
        if mode == 'train':
            labels = data['if_margin_trading'].astype(int)
            self._labelTensor = torch.tensor(labels, dtype=torch.int64)
        elif mode == 'infer':
            user_ids = data['user_id'].astype(int)
            self._labelTensor = torch.tensor(user_ids, dtype=torch.int64)
        
        feat_cols = ["net_investment_amount", "trading_amount", "num_buy_stocks", "num_sell_stocks", "trading_frequency"]
        raw_features = data[feat_cols].values
        features = np.apply_along_axis(row_parse, 1, raw_features).astype(float)
        self._featuresTensor = torch.tensor( features, dtype=torch.float32 )


    def __getitem__(self, index):
        return (self._featuresTensor[index], self._labelTensor[index])


    def __len__(self):
        return 100000



if __name__ == "__main__":
    path = "../../../../data/seq4rec/train_data.csv"
    data = seq4recDataset(path)
    print(data)

    # data = pd.read_csv(path, header=0, dtype=str)

    # labels = data.head()['if_margin_trading'].astype(int)
    # labelTensor = torch.tensor(labels, dtype=torch.int64)


    # feat_cols = ["net_investment_amount", "trading_amount", "num_buy_stocks", "num_sell_stocks", "trading_frequency"]
    # raw_features = data.head()[feat_cols].values
    
    
    # def row_parse(array:np.ndarray):
    #     cat_row = "|".join( array.tolist() )
    #     cat_row = cat_row.replace("\n", "")
    #     return np.array( cat_row.split("|") )
    
    # features = np.apply_along_axis(row_parse, 1, raw_features).astype(float)
    # featuresTensor = torch.tensor( features, dtype=torch.float32)

    # print(featuresTensor.shape)