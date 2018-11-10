#!/usr/bin/env python3
"""
Created on Fri Nov 2 2018
@author: J. Guymont and M. Mehdizadeh
"""
import numpy as np
import scipy
import copy
import time
import warnings

from mlp.nn import NN

class MLPClassifier(NN):
    """MLP classifier

    Simple MLP classifier implementation with one hidden layer and
    a relu activation function and a softmax output activation

    Args
        input_size: (int) dimension of the input
        hidden_size: (int) number of neurons of the hidden layer
        output_size: (int) number of classes
        regularization: (bool) should elastic net regularization be applied
    """

    DEFAULT_HYPERPARAMETERS = {
        'hidden_size': 2, 
        'learning_rate': 0.05, 
        'num_epochs': 100, 
        'lambda11': 0.001,
        'lambda12': 0.001,
        'lambda21': 0.001,
        'lambda22': 0.001
    }

    def __init__(self, input_size,  
                       output_size,
                       hidden_size=2, 
                       learning_rate=0.05, 
                       num_epochs=100, 
                       lambda11=0.001,
                       lambda12=0.001,
                       lambda21=0.001,
                       lambda22=0.001,
                       regularization=True,
        ):
        
        self.input_size = input_size 
        self.output_size = output_size
        self._regularization = regularization
        
        # hyperparameters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.lambda11 = lambda11
        self.lambda12 = lambda12
        self.lambda21 = lambda21
        self.lambda22 = lambda22

        self._param_names = {0:'W_xh', 1:'b_xh', 2:'W_hy', 3:'b_hy'}

        # parameters initialization
        self.W_xh = None
        self.b_xh = None
        self.W_hy = None
        self.b_hy = None
        self.initialize()

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
    
    def initialize(self):
        self.W_xh = self.uniform_initalization(shape=(self.hidden_size, self.input_size))
        self.b_xh = self.zero_initialization(shape=(self.hidden_size, 1))
        self.W_hy = self.uniform_initalization(shape=(self.output_size, self.hidden_size))
        self.b_hy = self.zero_initialization(shape=(self.output_size, 1))

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

    def forward_root1(self, x1, x2):
        """Forward propagation
        
        Args
            x: (array) array of dimension <n x d>

        return: (Array) Array of dimension <n x m> 
            where `m` is the number of class and `n` 
            is the number of examples. 
        """
        
        x = np.array([x1, x2])
        x = x.reshape(1, 2)
        return self.forward_root2(x2)

    def forward_root2(self, x2):
        x = np.array([x1, x2])
        x = x.reshape(1, 2)
        ha = self.W_xh.dot(x.T) + self.b_xh
        hs = self.relu(ha)
        oa = self.W_hy.dot(hs) + self.b_hy
        os = self.softmax(oa)
        out = os.reshape(2,) - np.array([0.5, 0.5])
        return out

    def backward(self, X, Y, crazy_loop=False):
        """Backward probagation

        Args
            X: (array) input batch of dimension <k x d>
            Y: (array) target batch of dimension <k x 1>
        """
        X = self._validate_input(X)
        
        batch_size = X.shape[0]

        self._reset_gradients()
        
        if not crazy_loop or X.shape[0] == 1:
            self.forward(X, train=True)
            grad_oa    = self.os - self._onehot(Y).T
            grad_hs    = grad_oa.T.dot(self.W_hy)
            grad_hs_ha = self._relu_prime(self.ha)       
            grad_ha    = grad_hs_ha * grad_hs.T          
            
            self.grad_W_hy  = grad_oa.dot(self.hs.T) / batch_size      
            self.grad_b_hy  = np.sum(grad_oa, axis=1).reshape(grad_oa.shape[0], 1) / batch_size
            self.grad_W_xh  = grad_ha.dot(X) / batch_size               
            self.grad_b_xh  = np.sum(grad_ha, axis=1).reshape(grad_ha.shape[0], 1) / batch_size
        else:
            for x, y in zip(X, Y):
                x = self._validate_input(x)
                self.forward(x, train=True)
                grad_oa    = self.os - self._onehot(y).T
                grad_hs    = grad_oa.T.dot(self.W_hy)
                grad_hs_ha = self._relu_prime(self.ha)       
                grad_ha    = grad_hs_ha * grad_hs.T          
                
                self.grad_W_hy  += grad_oa.dot(self.hs.T) / batch_size
                self.grad_b_hy  += grad_oa.reshape(grad_oa.shape[0], 1) / batch_size 
                self.grad_W_xh  += grad_ha.dot(x) / batch_size                
                self.grad_b_xh  += grad_ha.reshape(grad_ha.shape[0], 1) / batch_size

        assert(self.grad_W_hy.shape == self.W_hy.shape)
        assert(self.grad_b_hy.shape == self.b_hy.shape)
        assert(self.grad_W_xh.shape == self.grad_W_xh.shape)
        assert(self.grad_b_xh.shape == self.grad_b_xh.shape)

        if self._regularization:
            grad_reg_11 = self.lambda11 * np.sign(self.W_xh)
            grad_reg_12 = self.lambda12 * 2 * self.W_xh 
            grad_reg_21 = self.lambda21 * np.sign(self.W_hy)
            grad_reg_22 = self.lambda22 * 2 * self.W_hy 

            self.grad_W_hy += grad_reg_21 + grad_reg_22 
            self.grad_W_xh += grad_reg_11 + grad_reg_12

    def train(self, trainloader, devloader=None, testloader=None, crazy_loop=False):
        """Train the model using stochastic gradient"""
        starting_time = time.time()

        loss_storage = {'train': [], 'valid': [], 'test': []}
        acc_storage = {'train': [], 'valid': [], 'test': []}

        for epoch in range(self.num_epochs):    

            for inputs, targets in trainloader:

                self.backward(inputs, targets, crazy_loop)

                self.W_xh -= self.learning_rate * self.grad_W_xh 
                self.b_xh -= self.learning_rate * self.grad_b_xh 
                self.W_hy -= self.learning_rate * self.grad_W_hy 
                self.b_hy -= self.learning_rate * self.grad_b_hy 

            loss_storage['train'].append(self._nll_loss(trainloader))
            loss_storage['valid'].append(self._nll_loss(devloader))
            loss_storage['test'].append(self._nll_loss(testloader))

            acc_storage['train'].append(self.eval_accuracy(trainloader))
            acc_storage['valid'].append(self.eval_accuracy(devloader))
            acc_storage['test'].append(self.eval_accuracy(testloader))

            if epoch % (self.num_epochs // 10) == 0 and epoch > 0:
                current_time = time.time()
                loss = self._nll_loss(trainloader)
                train_acc = self.eval_accuracy(trainloader)
                dev_acc = self.eval_accuracy(devloader)
                print('epoch: {} | loss: {} | train acc: {}% | valid acc: {}% | time: {}'.format(
                    epoch, loss, train_acc, dev_acc, self._get_time(starting_time, current_time)
                ))
        return loss_storage, acc_storage

    def get_gradients(self):
        grads = []
        param_names = []
        for k, grad in enumerate(self.gradients()):
            for i in range(grad.shape[0]):
                for j in range(grad.shape[1]):
                    grads.append(grad[i,j])
                    param_names.append('{}_{}{}'.format(self._param_names[k], i, j))
        return grads, param_names
            
    def _get_time(self, starting_time, current_time):
        total_time = current_time - starting_time
        minutes = round(total_time // 60)
        seconds = round(total_time % 60)
        return '{} min., {} sec.'.format(minutes, seconds)
    
    def finite_difference_check(self, x, y, crazy_loop, eps=1e-5):
        """Finite difference gradient check
        
            gradient ~= (L(x, y; w + eps) - L(x, y; w)) / eps = gradHat

            Should have 0.99 < grad / gradHat < 1.01 
        """ 
        gradHats = []
        grads = []

        self.backward(x, y, crazy_loop)

        loss_raw = self.nll_loss(x, y)

        param_names = []
        k = -1

        for param, grad in zip(self.parameters(), self.gradients()):
            
            k += 1

            for i in range(param.shape[0]):
                
                for j in range(param.shape[1]):

                    param_names.append('{}_{}{}'.format(self._param_names[k], i, j))            
                    
                    param[i,j] += eps
                    loss_mod = self.nll_loss(x, y)
                    param[i,j] -= eps
                    
                    gradHat = (loss_mod - loss_raw) / eps
                    gradHats.append(gradHat)

                    cur_grad = grad[i,j]
                    grads.append(cur_grad)
                    
                    if grad[i,j] == 0 and gradHat == 0:
                        ratio = 1.
                    else:
                        ratio = (cur_grad) / (gradHat)
                    
                    if not 0.99 < ratio < 1.01:
                        # warnings.warn('Finite difference not valid: ratio = {}'.format(ratio), stacklevel=2)
                        print('Finite difference not valid: ratio = {}'.format(ratio))
                    else:
                        print('Finite difference check ok!')

        self._reset_activations()
        return gradHats, grads, param_names

    def nll_loss(self, x, y, reduc='mean'):
        """Negative loss likelihood

        Args
            x: (array) input
            y: (int) class in {0,...,m-1} where m is the number of classes
            regularization: (bool) should elastic net regularization be applied
        """
        prob = np.einsum('ky,ky->k', self.forward(x), self._onehot(y))
        loss = -np.log(prob)
        if self._regularization:
            loss += self.lambda11 * np.sum(np.abs(self.W_xh))
            loss += self.lambda12 * np.sum(np.square(self.W_xh))
            loss += self.lambda21 * np.sum(np.abs(self.W_hy))
            loss += self.lambda22 * np.sum(np.square(self.W_hy))
        if reduc == 'mean':
            loss = np.mean(loss) 
        elif reduc == 'sum': 
            loss = np.sum(loss)
        return loss

    def _nll_loss(self, dataloader):
        """Negative loss likelihood

        Args
            x: (array) input
            y: (int) class in {0,...,m-1} where m is the number of classes
        """
        if dataloader is None: return 'na'
        loss = 0
        for X, Y in dataloader:
            loss += self.nll_loss(X, Y, reduc='sum')
        loss = round(loss / dataloader.data_size_(), 5)
        return loss

    def predict(self, X):
        prob = self.forward(X)
        return np.argmax(prob.T, axis=0)

    def eval_accuracy(self, dataloader):
        if dataloader is None: return 'na'
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
