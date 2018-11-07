#!/usr/bin/env python3
"""
Created on Sat Aug 18 2018
@author: J. Guymont
"""

import os
import pickle
import numpy as np
import pandas as pd
import random

def normalize(x):
    """Normalize an array.
    Arguments:
        x: (Array) array representation of an image
    """
    mean = x.mean()
    std = x.std()
    return (x - mean)/std

def scale(x):
    """Scale all the pixel between -1 and 1.
    Arguments:
        x: (Array) array representation of an image
            i.e. a matrix of the same dimension as the
            original image with pixel value as element
    """
    min_value = np.min(x)
    max_value = np.max(x)
    scaled_image = (x - min_value)/(max_value - min_value)*2 - 1
    return scaled_image

def load_fashion_mnist_test(test_path, do_normalize=True):
    data_test = pd.read_csv(test_path)

    test_x = np.array(data_test.iloc[:, 1:]).reshape(10000, 28, 28).astype('float32')
    test_y = np.array(data_test.iloc[:, 0])

    if do_normalize:
        test_x = np.array([normalize(x) for x in test_x])

    return test_x, test_y

def load_fashion_mnist_train(train_path, split=[0.8, 0.2], do_normalize=True):

    data_train = pd.read_csv(train_path)

    X = np.array(data_train.iloc[:, 1:]).reshape(60000, 28, 28).astype('float32')
    
    if do_normalize:
        X = np.array([normalize(x) for x in X])

    y = np.array(data_train.iloc[:, 0])

    data_size = len(y)
    train_pct = split[0]

    train_size = round(data_size*train_pct)

    # A list of id (1,...,data_size)
    examples = range(data_size)

    # draw training index
    train_ix = random.sample(examples, train_size)

    #: remove training data index
    valid_ix = [example for example in examples if example not in train_ix]

    train_x = X[train_ix]
    train_y = y[train_ix]

    valid_x = X[valid_ix]
    valid_y = y[valid_ix]

    return train_x, train_y, valid_x, valid_y

if __name__ == '__main__':

    TRAIN_FILE = './data/mnist/train.csv'
    TEST_FILE = './data/mnist/test.csv'

    train_x, train_y, valid_x, valid_y = load_fashion_mnist_train(TRAIN_FILE, split=[0.8, 0.2], do_normalize=True)
    test_x, test_y = load_fashion_mnist_test(TEST_FILE, do_normalize=True)

    data = {
        "train_x": train_x,
        "train_y": train_y,
        "valid_x": valid_x,
        "valid_y": valid_y,
        "test_x":  test_x,
        "test_y":  test_y
    }

    with open('./data/mnist/normalized_data.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)