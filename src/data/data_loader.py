from typing import List

import random
import pandas
import numpy
import torch
from torch.utils.data import Dataset


class CtrDataset(Dataset):
    def __init__(self,
                 data: pandas.DataFrame,
                 numerical_cols: List[str],
                 categorical_cols: List[str],
                 salt:bool=True,
                 transform=None):
        super().__init__()
        self.data = data
        self.transform = transform
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.salt = salt

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx:int):
        line = self.data.iloc[idx,:]
        # map id=0 if id is NULL. that's why adding a aditional dimention.
        categoricals = line[self.categorical_cols].values.astype(numpy.int32) + 1
        if self.salt:
            categoricals = self.__random_noise(categoricals)
        label = line.clicked.astype(numpy.int32)
        if len(self.numerical_cols)!=0:
            numericals = line[self.numerical_cols].values.astype(numpy.float32).reshape(1,-1)
            return torch.FloatTensor(numericals), torch.LongTensor(categoricals), torch.LongTensor([label])
        else:
            return torch.FloatTensor([1.0]), torch.LongTensor(categoricals), torch.LongTensor([label])
    
    def __random_noise(self, cats:numpy.ndarray, threshold:float=0.75):
        for i in range(cats.shape[0]):
            if random.random()>threshold:
                cats[i] = 0
        return cats


def loader(dataset:Dataset, batch_size:int, shuffle:bool=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4)
    return loader