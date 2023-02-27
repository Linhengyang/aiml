import torch
import pandas as pd
import numpy as np

def id2index(idx):
    return int(idx) - 1

def newest_mask(df):
    df['newest_mask'] =  df['timestamp'] != df['timestamp'].max()
    return df

class MovieLensRatingDataset(torch.utils.data.Dataset):
    def __init__(self, path, is_train, split_method='time-aware', test_ratio = 0.1, seed=1026):
        super().__init__()
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_csv(path, '\t', names=names, engine='python')
        self._num_users = data.user_id.unique().shape[0]
        self._num_items = data.item_id.unique().shape[0]
        data['user_id'] = data['user_id'].map(id2index)
        data['item_id'] = data['item_id'].map(id2index)
        if split_method == 'time-aware':
            data.sort_values(['user_id', 'timestamp'], ascending=[True, True], inplace=True)
            data = data.groupby('user_id').apply(newest_mask)
            mask = data.newest_mask.to_list()
            data.drop('newest_mask', axis=1, inplace=True)
        elif split_method == 'random':
            np.random.seed(seed) # 固定随机seed, 以保证trainset和testset是互补的
            mask = np.random.uniform(0, 1, len(data)) < 1 - test_ratio
        if is_train:
            self._split_data = data[mask]
        else:
            self._split_data = data[~mask]
        self.user_tensor = torch.tensor(self._split_data.user_id.to_list(), dtype=torch.int64)
        self.item_tensor = torch.tensor(self._split_data.item_id.to_list(), dtype=torch.int64)
        self.score_tensor = torch.tensor(self._split_data.rating.to_list(), dtype=torch.float32)
        self._interactions_itembased = torch.zeros(self._num_items, self._num_users, dtype=torch.float32)
        self._interactions_itembased[self.item_tensor, self.user_tensor] = self.score_tensor
        self._interactions_userbased = self._interactions_itembased.transpose(1, 0)
    
    def __getitem__(self, index):
        return (self.user_tensor[index], self.item_tensor[index], self.score_tensor[index])
    
    def __len__(self):
        return self._split_data.shape[0]
    
    @property
    def num_users(self):
        return self._num_users
    
    @property
    def num_items(self):
        return self._num_items
    
    @property
    def dataframe(self):
        return self._split_data
    
    @property
    def interactions_itembased(self):
        return self._interactions_itembased

    @property
    def interactions_userbased(self):
        return self._interactions_userbased