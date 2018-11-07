#!/usr/bin/env python3
"""
Created on Fri Nov 2 2018
@author: J. Guymont and M. Mehdizadeh
"""
import numpy as np

from nn import NN

class MLPClassifier(NN):
    """MLP classifier

    Simple MLP classifier implementation with one hidden layer and
    a relu activation function and a softmax output activation

    Args
        input_size: (int) dimension of the input
        hidden_size: (int) number of neurons of the hidden layer
        output_size: (int) number of classes
    """

    def __init__(self, input_size, hidden_size, ouput_size):
        self.input_size = input_size 
        self.ouput_size = ouput_size
        self.hidden_size = hidden_size
        self.W1 = self.uniform_initalization(shape=(hidden_size, input_size))
        self.b1 = self.zero_initialization(shape=(hidden_size, 1))
        self.W2 = self.uniform_initalization(shape=(ouput_size, hidden_size))
        self.b2 = self.zero_initialization(shape=(ouput_size, 1))

    def forward(self, x):
        """Forward propagation
        
        Args
            x: (array) array of dimension <n x d>
        """
        try:
            d = x.shape[1]
        except IndexError:
            d = x.shape[0]
            x = x.reshape(1, d)
        ha = self.W1.dot(x.T) + self.b1
        hs = self.relu(ha)
        oa = self.W2.dot(hs) + self.b2
        os = self.softmax(oa)
        return os

    def backward(self, x, y):
        """Backward probagation
        """
        x = x.reshape(1, self.input_size)

        # forward propagation
        ha = self.W1.dot(x.T) + self.b1
        hs = self.relu(ha)
        oa = self.W2.dot(hs) + self.b2
        os = self.softmax(oa)
        
        # gradient of L wrt W2
        grad_oa = os - self._onehot(y) # <m x 1>
        grad_w2 = grad_oa.dot(hs.T)    # <m x H>
        
        # gradient of L wrt b2
        grad_b2 = grad_oa              # <m x 1>

        # gradient of L wrt W1
        grad_hs_ha = np.diag(self._relu_derivative(ha).reshape(self.hidden_size,))
        grad_ha = grad_oa.T.dot(self.W2).dot(grad_hs_ha) 
        grad_w1 = grad_ha.T.dot(x)

        # gradient of L wrt b1
        grad_b1 = grad_ha.T

    def nll_loss(self, x, y):
        """Negative loss likelihood

        Args
            x: (array) input
            y: (int) class \in {0,...,m-1} where m is the number of classes
        """
        prob = self.forward(x).dot(self._onehot(y))
        return -np.log(prob)

    def _relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def _onehot(self, y):
        onehot = np.zeros((self.ouput_size, 1))
        onehot[y, 0] = 1
        return onehot

if __name__ == '__main__':
    
    x1 = [1, 2, 3, 4]
    x2 = [2,-1, 5, 3]
    x = np.array([x1, x2]) # 2 x 4
    y = np.array([0, 2, 1])

    mlp = MLPClassifier(input_size=4, hidden_size=2, ouput_size=3)
    pred = mlp.forward(x[0])
    mlp.backward(x[0], y[0])

