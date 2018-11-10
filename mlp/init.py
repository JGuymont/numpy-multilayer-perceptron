#!/usr/bin/env python3
"""
Created on Fri Nov 2 2018
@author: J. Guymont and M. Mehdizadeh
"""

import numpy as np

class Init:

    @staticmethod
    def uniform_initalization(shape):
        """Glorot (uniform) parameter initialization
        
        Arguments
            shape: (tuple) dimension of the parameter input_size x output_size
        """
        input_size = shape[0]
        return np.random.uniform(low=-1.0/np.sqrt(input_size), high=1.0/np.sqrt(input_size), size=shape)

    @staticmethod
    def zero_initialization(shape):
        return np.zeros(shape)

if __name__ == '__main__':
    init = Init()
    w = init.uniform_initalization(shape=(2,3))
    print(w)    
    b = init.zero_initialization(shape=3)
    print(b)
