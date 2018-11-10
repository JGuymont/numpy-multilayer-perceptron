import numpy as np
import random
import time

from mlp.mlp import MLPClassifier
from mlp.dataloader import DataLoader

np.random.seed(0)

CIRCLE_DATA_PATH = './data/circles/circles.txt'

class Data:
    """Abstract class for the circles dataset

    Args
        path: (string) path to the dataset
        input_dim: (int)
        split: (list) list of float [train_pct, valid_pct, test_pct]
    """

    def __init__(self, path, input_dim, split):
        self._raw_data = np.loadtxt(open(path, 'r'))
        self._input_dim = input_dim
        self._data = self._read_data()
        self._data_index = self._split_index(split)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def _read_data(self):
        inputs = self._raw_data[:, :self._input_dim]
        targets = self._raw_data[:, -1].astype(int)
        return list(zip(inputs, targets))

    def _split_index(self, split):
        storage = {'train': [], 'valid': [], 'test': []}
        train_size = round(len(self)*split[0])
        valid_size = round((len(self) - train_size)*split[1])

        examples = range(len(self))
        storage['train'] = random.sample(examples, train_size)
        examples = [ex for ex in examples if ex not in storage['train']] # remove index
        storage['valid'] = random.sample(examples, valid_size)
        storage['test'] = [ex for ex in examples if ex not in storage['valid']]
        return storage

    def train(self):
        return [self._data[i] for i in self._data_index['train']]

    def valid(self):
        return [self._data[i] for i in self._data_index['valid']]

    def test(self):
        return [self._data[i] for i in self._data_index['test']]

    def dim_(self):
        return self._input_dim

if __name__ == '__main__':

    INPUT_DIM = 2
    OUTPUT_DIM = 2

    # hyperparameters
    HIDDEN_DIM = 2
    BATCH_SIZE = 10
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.01

    data = Data(CIRCLE_DATA_PATH, input_dim=2, split=[0.7, 0.15, 0.15])
    trainloader = DataLoader(data.train(), batch_size=BATCH_SIZE)
    devloader = DataLoader(data.valid(), batch_size=BATCH_SIZE)

    mlp = MLPClassifier(
        input_size=INPUT_DIM, 
        hidden_size=HIDDEN_DIM, 
        output_size=OUTPUT_DIM, 
        learning_rate=LEARNING_RATE, 
        num_epochs=NUM_EPOCHS
    )

    # mlp.train(trainloader, devloader, crazy_loop=True)

    def question1():
        x = data[0]
        y = data[0]
        trainloader = DataLoader(data[0], batch_size=BATCH_SIZE)
        mlp.train(trainloader, crazy_loop=True)
        mlp._numerical_gradient(trainloader)

    question1()