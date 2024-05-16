import torch
import pandas as pd
import numpy as np

def row_parse(array:np.ndarray):
    cat_row = "|".join( array.tolist() )
    cat_row = cat_row.replace("\n", "")
    return np.array( cat_row.split("|") )



class seq4recDataset(torch.utils.data.Dataset):
    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data = pd.read_csv(path, header=0, dtype=str)
        labels = data['if_margin_trading'].astype(int)
        self._labelTensor = torch.tensor(labels, dtype=torch.int64)
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