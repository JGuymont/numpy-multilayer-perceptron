#!/usr/bin/env python3
"""
Created on Mon Jan 15 13:31:59 2018
@author: J. Guymont
"""
import numpy as np

class DataLoader:

    def __init__(self, data, batch_size):
        self._data = data
        try:
            self._inputs, self._targets = [list(t) for t in zip(*data)] 
        except TypeError:
            self._inputs = [data[0]]
            self._targets = [data[1]]
        self._batch_size = batch_size
        self._data_size = len(self._data)
        self._n_batch = self._data_size // self._batch_size + 1

        self._dataloader = self._split_in_batch()

    def __len__(self):
        return self._n_batch

    def __getitem__(self, i):
        return self._dataloader[i]

    def _get_next_input_batch(self, last=False):
        last_idx = self._batch_size if not last else len(self._inputs)
        cur_batch = np.array([self._inputs.pop(0) for _ in range(last_idx)])
        return cur_batch

    def _get_next_target_batch(self, last=False):
        last_idx = self._batch_size if not last else len(self._targets)
        cur_batch = np.array([self._targets.pop(0) for _ in range(last_idx)])
        return cur_batch

    def _split_in_batch(self):
        storage = {'inputs': [], 'targets': []}
        for i in range(self._n_batch-1):
            storage['inputs'].append(self._get_next_input_batch())
            storage['targets'].append(self._get_next_target_batch())
        if not len(self._inputs) == 0:
            storage['inputs'].append(self._get_next_input_batch(last=True))
            storage['targets'].append(self._get_next_target_batch(last=True)) 
        return list(zip(storage['inputs'], storage['targets']))

    def data_size_(self):
        return self._data_size