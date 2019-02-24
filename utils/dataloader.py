#!/usr/bin/env python3
"""
Created on Mon Jan 15 13:31:59 2018
@author: J. Guymont
"""
import random
import numpy as np
from collections import Iterator


class DataLoader(Iterator):

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batch = len(dataset) // batch_size + len(dataset) % batch_size
        self.examples = self.partitionize(list(range(len(self.dataset))))

    def __next__(self):
        if not self.examples:
            self.examples = self.partitionize(list(range(len(self.dataset))))
            raise StopIteration
        return self.next_batch()

    def __len__(self):
        return self.n_batch

    def next_batch(self):
        batch = [self.dataset[i] for i in self.examples.pop()]
        x, y = zip(*batch)
        x = np.array(x)
        y = np.array(y)
        return x, y

    def partitionize(self, examples):
        examples = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(examples)
        return [examples[i::self.n_batch] for i in range(self.n_batch)]
     