#!/usr/bin/env python3
"""
Created on Fri Nov 2 2018
@author: J. Guymont and M. Mehdizadeh
"""
import numpy as np
import copy
import time

from mlp.nn import NN

class MLPClassifier(NN):
    """MLP classifier

    Simple MLP classifier implementation with one hidden layer and
    a relu activation function and a softmax output activation

    Args
        input_size: (int) dimension of the input
        hidden_size: (int) number of neurons of the hidden layer
        output_size: (int) number of classes
    """

    def __init__(self, input_size, 
                       hidden_size, 
                       output_size, 
                       learning_rate, 
                       num_epochs, 
                       lambda11=0.01,
                       lambda12=0.01,
                       lambda21=0.01,
                       lambda22=0.01
        ):
        
        self.input_size = input_size 
        self.output_size = output_size
        
        # hyperparameters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.lambda11 = lambda11
        self.lambda12 = lambda12
        self.lambda21 = lambda21
        self.lambda22 = lambda22

        # parameters initialization
        self.W_xh = self.uniform_initalization(shape=(hidden_size, input_size))
        self.b_xh = self.zero_initialization(shape=(hidden_size, 1))
        self.W_hy = self.uniform_initalization(shape=(output_size, hidden_size))
        self.b_hy = self.zero_initialization(shape=(output_size, 1))

        # initialize gradient to zero
        self.grad_W_xh = 0
        self.grad_b_xh = 0
        self.grad_W_hy = 0
        self.grad_b_hy = 0

        # we will keep activations for the backward propagations
        self.ha = None
        self.hs = None
        self.oa = None
        self.os = None

        # we will keep the gradients that are use 
        # to compute the gradient of more then one 
        # parameters
        self.grad_oa = None
        self.grad_ha = None

    def parameters(self):
        return [self.W_xh, self.b_xh, self.W_hy, self.b_hy]

    def gradients(self):
        return [self.grad_W_xh, self.grad_b_xh, self.grad_W_hy, self.grad_b_hy]

    def forward(self, x, train=False):
        """Forward propagation
        
        Args
            x: (array) array of dimension <n x d>

        return: (Array) Array of dimension <n x m> 
            where `m` is the number of class and `n` 
            is the number of examples. 
        """
        x = self._validate_input(x)
        ha = self.W_xh.dot(x.T) + self.b_xh
        hs = self.relu(ha)
        oa = self.W_hy.dot(hs) + self.b_hy
        os = self.softmax(oa)
        if train:
            self.ha, self.hs, self.oa, self.os = ha, hs, oa, os
        return os.T

    def backward(self, X, Y, crazy_loop=False):
        """Backward probagation

        Args
            X: (array) input batch of dimension <k x d>
            Y: (array) target batch of dimension <k x 1>
        """
        X = self._validate_input(X)
        
        batch_size = X.shape[0]
        
        self.forward(X, train=True)

        grad_oa    = self.os - self._onehot(Y).T
        grad_W_hy  = grad_oa.dot(self.hs.T)       
        grad_b_hy  = np.sum(grad_oa, axis=1).reshape(grad_oa.shape[0], 1)
        grad_hs    = grad_oa.T.dot(self.W_hy)
        grad_hs_ha = self._relu_prime(self.ha)       
        grad_ha    = grad_hs_ha * grad_hs.T          
        grad_W_xh  = grad_ha.dot(X)                
        grad_b_xh  = np.sum(grad_ha, axis=1).reshape(grad_ha.shape[0], 1)
        
        assert(grad_W_hy.shape == self.W_hy.shape)
        assert(grad_b_hy.shape == self.b_hy.shape)
        assert(grad_W_xh.shape == self.W_xh.shape)
        assert(grad_b_xh.shape == self.b_xh.shape)

        grad_reg_11 = np.mean(np.sign(self.W_xh))
        grad_reg_12 = 2 * sum(self.W_xh)
        grad_reg_21 = np.mean(np.sign(self.W_hy))
        grad_reg_22 = 2 * sum(self.W_hy)

        self.grad_W_hy = grad_W_hy / batch_size + self.lambda21 * grad_reg_21 + self.lambda22 * grad_reg_22 
        self.grad_b_hy = grad_b_hy / batch_size
        self.grad_W_xh = grad_W_xh / batch_size + self.lambda11 * grad_reg_11 + self.lambda12 * grad_reg_12
        self.grad_b_xh = grad_b_xh / batch_size

    def train(self, trainloader, devloader=None, crazy_loop=False):
        """Train the model using stochastic gradient"""
        starting_time = time.time()
        for epoch in range(self.num_epochs):    

            for inputs, targets in trainloader:

                self.backward(inputs, targets, crazy_loop)

                self.W_xh -= self.learning_rate * self.grad_W_xh 
                self.b_xh -= self.learning_rate * self.grad_b_xh 
                self.W_hy -= self.learning_rate * self.grad_W_hy 
                self.b_hy -= self.learning_rate * self.grad_b_hy 

            if epoch % 10 == 0 and epoch > 0:
                current_time = time.time()
                loss = self._nll_loss(trainloader)
                train_acc = self.eval_accuracy(trainloader)
                dev_acc = self.eval_accuracy(devloader) if devloader is not None else 'na'
                print('epoch: {} | loss: {} | train acc: {}% | valid acc: {}% | time: {}'.format(
                    epoch, loss, train_acc, dev_acc, self._get_time(starting_time, current_time)
                ))
                
    def _get_time(self, starting_time, current_time):
        total_time = current_time - starting_time
        minutes = round(total_time // 60)
        seconds = round(total_time % 60)
        return '{} min., {} sec.'.format(minutes, seconds)
    
    def finite_difference_check(self, x, y, eps=1e-5):
        """Finite difference gradient check
        
            gradient ~= (L(x, y; w + eps) - L(x, y; w)) / eps = gradHat

            Should have 0.99 < grad / gradHat < 1.01 
        """ 
        self.forward(x, train=True)
        self.backward(x, y)
        loss1 = self.nll_loss(x, y)
        for param, grad in zip(self.parameters(), self.gradients()):
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    param[i,j] += eps
                    loss2 = self.nll_loss(x, y)
                    param[i,j] -= eps
                    gradHat = (loss2 - loss1) / eps
                    ratio = (grad[i,j]+1e-10) / (gradHat[0][0]+1e-10)
                    if not 0.99 < ratio < 1.01:
                        raise Warning('Finite difference not valid: ratio = {}'.format(ratio))
        self._reset_activations()

    def nll_loss(self, X, Y):
        """Negative loss likelihood

        Args
            x: (array) input
            y: (int) class in {0,...,m-1} where m is the number of classes
        """
        loss = 0
        for x, y in zip(X, Y):
            prob = self.forward(x).dot(self._onehot(y).T)
            loss += -np.log(prob)
        return loss / len(y)
    
    def _nll_loss(self, dataloader):
        """Negative loss likelihood

        Args
            x: (array) input
            y: (int) class in {0,...,m-1} where m is the number of classes
        """
        loss = 0
        for X, Y in dataloader:
            for x, y in zip(X, Y):
                prob = self.forward(x).dot(self._onehot(y).T)
                loss += -np.log(prob)
                loss += self.lambda11 * np.sum(np.abs(self.W_xh))
                loss += self.lambda12 * np.sum(np.square(self.W_xh))
                loss += self.lambda21 * np.sum(np.abs(self.W_hy))
                loss += self.lambda22 * np.sum(np.square(self.W_hy))
                
        loss = round(loss[0][0] / dataloader.data_size_(), 5)
        return loss

    def predict(self, X):
        prob = self.forward(X)
        return np.argmax(prob.T, axis=0)

    def eval_accuracy(self, dataloader):
        correct, total = 0, 0
        for inputs, targets in dataloader:
            y_hat = self.predict(inputs)
            correct += (y_hat == targets).sum()
            total += len(targets)
        acc = correct / total
        return round(acc*100, 2)

    def _validate_input(self, x):
        """Make sure the input have the dimension <n x d>
        where n is the size of a batch"""
        try:
            d = x.shape[1]
            if not d == self.input_size:
                raise ValueError('The dimension of x should be {} not {}'.format(self.input_size, d))
        except IndexError:
            d = x.shape[0]
            if not d == self.input_size:
                raise ValueError('The dimension of x should be {} not {}'.format(self.input_size, d))
            x = x.reshape(1, d)
        return x

    def _relu_prime(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def _onehot(self, y):
        """convert an integer into a onehot vector 
        of dimension <m x n> where m is the number
        of classes
        
        Args
            y: (array) Array (of int) with dim <n> ((n, ) in numpy format)
                where `n` is the number of training example, 
                each element correspondind to a target

        return: (Array) Array of dimension <n x m> 
        """
        
        try:
            n = y.shape[0]
        except IndexError:
            n = 1
            onehot = np.zeros((1, self.output_size))
            onehot[0, y] = 1
            return onehot
        onehot = np.zeros((n, self.output_size))
        onehot[np.arange(n), y] = 1
        return onehot

    def _reset_activations(self):
        self.ha = None
        self.hs = None
        self.oa = None
        self.os = None

    def _reset_gradients(self):
        self.grad_W_xh = 0
        self.grad_b_xh = 0
        self.grad_W_hy = 0
        self.grad_b_hy = 0

if __name__ == '__main__':
    # Very simple unit testing
    
    x1 = [1, 2, 3, 4]
    x2 = [2,-1, 5, 3]
    x = np.array([x1, x2]) # 2 x 4
    y = np.array([0, 1])

    mlp = MLPClassifier(input_size=4, hidden_size=2, output_size=3, 
                        learning_rate=0.001, num_epochs=10)
    
    pred = mlp.forward(x)
    mlp.backward(x, y)
    mlp.finite_difference_check(x[0], y[0])

