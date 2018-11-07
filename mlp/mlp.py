#!/usr/bin/env python3
"""
Created on Fri Nov 2 2018
@author: J. Guymont and M. Mehdizadeh
"""
import numpy as np
import copy

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

    def __init__(self, input_size, hidden_size, ouput_size, learning_rate, num_epochs):
        self.input_size = input_size 
        self.ouput_size = ouput_size
        
        # hyperparameters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # parameters initialization
        self.W_xh = self.uniform_initalization(shape=(hidden_size, input_size))
        self.b_xh = self.zero_initialization(shape=(hidden_size, 1))
        self.W_hy = self.uniform_initalization(shape=(ouput_size, hidden_size))
        self.b_hy = self.zero_initialization(shape=(ouput_size, 1))

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

    def backward(self, X, Y):
        """Backward probagation

        Args
            X: (array) input batch of dimension <k x d>
            Y: (array) target batch of dimension <k x 1>
        """
        X = self._validate_input(X)
        Y = [Y] if Y.shape == () else Y # in case there is only one target
        batch_size = X.shape[0]
        
        self._reset_gradients()
        for x, y in zip(X, Y):
            self.forward(x, train=True)
            x = x.reshape(1, self.input_size)
            self.grad_W_hy += self._get_gradient_W_hy(y)
            self.grad_b_hy += self._get_gradient_b_hy()
            self.grad_W_xh += self._get_gradient_W_xh(x)
            self.grad_b_xh += self._get_gradient_b_xh()

        for grad in self.gradients():
            grad /= batch_size 

        self._reset_activations()

    def train(self, dataloader):
        """Train the model using stochastic gradient"""
        for epoch in range(self.num_epochs):

            for inputs, targets in dataloader:

                self.backward(inputs, targets)

                self.W_xh -= self.learning_rate * self.grad_W_xh 
                self.b_xh -= self.learning_rate * self.grad_b_xh 
                self.W_hy -= self.learning_rate * self.grad_W_hy 
                self.b_hy -= self.learning_rate * self.grad_b_hy 

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
            
    def nll_loss(self, x, y):
        """Negative loss likelihood

        Args
            x: (array) input
            y: (int) class in {0,...,m-1} where m is the number of classes
        """
        prob = self.forward(x).dot(self._onehot(y).T)
        return -np.log(prob)
    
    def predict(self, X):
        prob = self.forward(X)
        return np.argmax(prob.T, axis=0)

    def accuracy(self, X, y):
        y_hat = self.predict(X)
        correct = (y_hat == y).sum()
        acc = correct / len(y)
        return round(acc*100, 4)

    def _validate_input(self, x):
        """Make sure the input have the dimension <n x d>
        where n is the size of a batch"""
        try:
            d = x.shape[1]
            if not d == self.input_size:
                raise ValueError('The dimension of x should be {}'.format(self.input_size))
        except IndexError:
            d = x.shape[0]
            if not d == self.input_size:
                raise ValueError('The dimension of x should be {}'.format(self.input_size))
            x = x.reshape(1, d)
        return x
    
    def _get_gradient_W_hy(self, y):
        """Gradient of the loss w.r.t W_hy. 
        
        Should have shape <m x d_h> where `m` is the 
        number of classes and d_h is the size of the 
        hidden layer
        """
        self.grad_oa = self.os - self._onehot(y).T
        grad_W_hy = self.grad_oa.dot(self.hs.T)
        return grad_W_hy

    def _get_gradient_b_hy(self):
        """Gradient of the loss w.r.t b_hy. 
        
        Should have shape <m x 1> where `m` is the 
        number of classes.
        """
        return self.grad_oa

    def _get_gradient_W_xh(self, x):
        """Gradient of the loss w.r.t W_xh. 
        
        Should have shape <d_h x d> where `d` is the 
        dimension of the input and `d_h` is the dimension 
        of the hidden layer.
        """
        grad_hs_ha = np.diag(self._relu_derivative(self.ha).reshape(self.hidden_size,))
        self.grad_ha = self.grad_oa.T.dot(self.W_hy).dot(grad_hs_ha) 
        grad_W_xh = self.grad_ha.T.dot(x)
        return grad_W_xh

    def _get_gradient_b_xh(self):
        """Gradient of the loss w.r.t b_xh. 
        
        Should have shape <d_h x 1> where `d_h` is the 
        dimension of the hidden layer.
        """
        return self.grad_ha.T

    def _relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
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
            onehot = np.zeros((1, self.ouput_size))
            onehot[0, y] = 1
            return onehot
        onehot = np.zeros((n, self.ouput_size))
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

    mlp = MLPClassifier(input_size=4, hidden_size=2, ouput_size=3, 
                        learning_rate=0.001, num_epochs=10)
    
    pred = mlp.forward(x)
    mlp.backward(x, y)
    mlp.finite_difference_check(x[0], y[0])

