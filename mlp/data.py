import numpy as np
import pickle

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
        _data_size = len(targets)
        _examples = range(_data_size)

        dataloader = []

        for _ in range(int(_data_size/batch_size)):

            #: randomly select examples for current SGD iteration
            _mini_batch = random.sample(_examples, batch_size)

            #: remove current example from the list of examples
            _examples = [example for example in _examples if example not in _mini_batch]

            #: Convert array to tensor of size [batch_size, 1, img_size, img_size]
            _batch_x = inputs[_mini_batch, :])).view(batch_size, 1, 28, 28)
            _batch_y = Variable(torch.LongTensor(targets[_mini_batch]))

            dataloader.append((_batch_x, _batch_y))

        return dataloader

if __name__ == '__main__':

    NORMALIZED_DATA_PATH = './data/mnist/normalized_data.pkl'

    with open(NORMALIZED_DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    trainloader = Dataloader().data_loader(data['train_x'], data['train_y'], batch_size=64)
    devloader = Dataloader().data_loader(data['valid_x'], data['valid_y'], batch_size=1000)
    testloader = Dataloader().data_loader(data['test_x'], data['test_y'], batch_size=1000) 

    dataloaders = {
        "trainloader": trainloader,
        "devloader": devloader,
        "testloader": testloader,
    }

    with open('./data/dataloaders.pkl', 'wb') as f:
        pickle.dump(dataloaders, f, pickle.HIGHEST_PROTOCOL)
