#!/usr/bin/env python3
"""
Created on Mon Jan 15 13:31:59 2018
@author: J. Guymont
"""

import random

class Dataloader:
    """
    Data loader. Combines a dataset and a sampler, and
    provides an iterators over the training dataset.
    Args:
        batch_size (int, optional): how many samples per batch to load (default: 1).
    """
    def __init__(self):
        pass

    def data_loader(self, inputs, targets, batch_size):
        """provides an iterator over a dataset"""
        data_size = len(targets)
        examples = range(data_size)

        dataloader = []

        for _ in range(int(data_size/batch_size)):

            #: randomly select examples for current SGD iteration
            mini_batch = random.sample(examples, batch_size)

            #: remove current example from the list of examples
            examples = [example for example in examples if example not in mini_batch]

            #: Convert array to tensor of size [batch_size, 1, img_size, img_size]
            batch_x = inputs[mini_batch, :].reshape(batch_size, 28, 28)
            batch_y = targets[mini_batch]

            dataloader.append((batch_x, batch_y))

        return dataloader