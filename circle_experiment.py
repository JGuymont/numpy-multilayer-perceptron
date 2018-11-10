import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import random
import time

from mlp.mlp import MLPClassifier
from mlp.dataloader import DataLoader

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

def plot_gradient(grad1, grad2, param_names, legend, title):
    plt.rcParams.update({'font.size': 6})
    plt.plot(param_names, grad1, '--')
    plt.plot(param_names, grad2, 'o')
    plt.legend(legend)
    plt.xlabel('parameter')
    plt.ylabel('gradient')
    plt.savefig(title)
    plt.show()

if __name__ == '__main__':

    INPUT_DIM = 2
    OUTPUT_DIM = 2

    data = Data(CIRCLE_DATA_PATH, input_dim=2, split=[0.7, 0.15, 0.15])

    def question12():
        batch = DataLoader(data.train(), batch_size=1)
        X = batch[0][0]
        Y = batch[0][1]
        
        mlp = MLPClassifier(INPUT_DIM, OUTPUT_DIM)
        gradHats, grads, param_names = mlp.finite_difference_check(X, Y, crazy_loop=True)

        plot_gradient(
            gradHats, grads, 
            param_names, 
            legend=['finite differences approx.', 'backpropagation'],
            title='plots/question2.jpg'
        )
        
    def question4():
        batch = DataLoader(data.train(), batch_size=10)
        X = batch[20][0]
        Y = batch[20][1]

        mlp = MLPClassifier(INPUT_DIM, OUTPUT_DIM)
        gradHats, grads, param_names = mlp.finite_difference_check(X, Y, crazy_loop=True)

        plot_gradient(
            gradHats, grads, 
            param_names, 
            legend=['finite differences approx.', 'backpropagation'],
            title='plots/question4.jpg'
        )

    def question5():

        def get_boudary():
            mlp = MLPClassifier(INPUT_DIM, OUTPUT_DIM)
            data = np.loadtxt(open(CIRCLE_DATA_PATH, 'r'))[:, :2]
            
            boundary_x = []
            boundary_y = []

            for x in data:
                sol = opt.root(mlp.forward_root, x)
                boundary_x.append(sol['x'][0])
                boundary_y.append(sol['x'][1])
            
            plt.scatter(boundary_x, boundary_y)
            plt.show()

        raw_data = np.loadtxt(open(CIRCLE_DATA_PATH, 'r'))[:, :2]

        # hyperparameters
        BATCH_SIZE = 32
        NUM_EPOCH = 1000

        HIDDEN_DIM_SET = [10]
        LEARNING_RATE_SET = [0.05]
        L1_WEIGH_DECAY = [0.001]
        L2_WEIGH_DECAY = [0.001]
        
        trainloader = DataLoader(data.train(), batch_size=BATCH_SIZE)
        devloader = DataLoader(data.valid(), batch_size=len(data.valid()))

        for h in HIDDEN_DIM_SET:
            for lr in LEARNING_RATE_SET:
                for l1 in L1_WEIGH_DECAY:
                    for l2 in L2_WEIGH_DECAY:
                
                        print('\nhidden_dim: {}, lr: {}, l1: {}, l2: {}'.format(h, lr, l1, l2))
                        mlp = MLPClassifier(INPUT_DIM, OUTPUT_DIM, h, lr, NUM_EPOCH, l1, l2, l1, l2)
                        mlp.train(trainloader, devloader, crazy_loop=False)

                        boundary_x1 = list(np.arange(-1, 1, 0.01))
                        boundary_x2 = []

                        for x1 in boundary_x1:
                            
                            sol = opt.root(mlp.forward_root, 0, x1=x1)
                            boundary_x2.append(sol['x'])
        
                        plt.scatter(boundary_x1, boundary_x2)
        plt.show()

    def question7():

        trainloader = DataLoader(data.train(), batch_size=1)

        #print(*trainloader[0])


        mlp = MLPClassifier(INPUT_DIM, OUTPUT_DIM)
        
        mlp.backward(*trainloader[0], crazy_loop=False)
        grads, param_names = mlp.get_gradients()

        mlp.backward(*trainloader[0], crazy_loop=True)
        grad_loop, param_names = mlp.get_gradients()

        plot_gradient(
            grads, grad_loop, 
            param_names, 
            legend=['Matrix calculus', 'Loop'],
            title='plots/question71.jpg'
        )

        trainloader = DataLoader(data.train(), batch_size=10)
        mlp = MLPClassifier(INPUT_DIM, OUTPUT_DIM)
        
        mlp.backward(*trainloader[0], crazy_loop=False)
        grads, param_names = mlp.get_gradients()

        mlp.backward(*trainloader[0], crazy_loop=True)
        grad_loop, param_names = mlp.get_gradients()

        plot_gradient(
            grads, grad_loop, 
            param_names, 
            legend=['Matrix calculus', 'Loop'],
            title='plots/question72.jpg'
        )

    #question12()
    #question4()
    #question5()
    #question7()
    