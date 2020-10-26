import numpy as np
import pandas as pd
import torch
import os
import json
import pickle
from functools import partial


def load_data(part):
    data = {}
    # data astnn
    root = 'data/poj_cls/'
    data_ = pd.read_pickle(root + f'{part}/blocks.pkl')
    data['astnn'] = data_

    # data code2vec
    with open(root + f'{part}/data.pkl', 'rb') as f:
        data_ = pickle.load(f)
    data['code2vec'] = data_

    return data


def get_batch_pc(data, idx, batch_size, max_context=600, pad_idx=0):
    x = data[idx: idx + batch_size]
    # reform x
    x_r = []
    for c in x:
        c = c[:max_context] # trimming (consider other strategies)
        c += [[pad_idx, pad_idx, pad_idx]] * (max_context - len(c)) # padding
        x_r.append(c)
    return torch.LongTensor(x_r).cuda()


def get_batch_st(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data = []
    for _, item in tmp.iterrows():
        data.append(item[2])
    return data


class DataLoader:
    "Temporary data loader for pretext task"

    def __init__(self, part, batch_size):
        self.data = load_data(part)
        self.batch_size = batch_size
    
    def __iter__(self):
        return ParallelIterator(
            self.data['astnn'], get_batch_st,
            self.data['code2vec'], get_batch_pc,
            self.batch_size
        )


class ParallelIterator:

    def __init__(self, x_a, handle_a, x_b, handle_b, batch_size):
        self.x_a = x_a
        self.handle_a = handle_a
        self.x_b = x_b
        self.handle_b = handle_b
        self.id = 0
        self.batch_size = batch_size
        assert len(x_a) == len(x_b)
        self.max_size = len(x_a)
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.id >= self.max_size:
            raise StopIteration
        else:
            mnb_x_a = self.handle_a(self.x_a, self.id, self.batch_size)
            mnb_x_b = self.handle_b(self.x_b, self.id, self.batch_size)
            self.id += self.batch_size
            return mnb_x_a, mnb_x_b


if __name__ == '__main__':
    dl = DataLoader(batch_size=64)
    for x1, x2 in dl.__iter__('train'):
        print(len(x1), len(x2))
