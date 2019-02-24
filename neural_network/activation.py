import numpy as np
from numpy import exp


class Activation:

    @staticmethod
    def softmax(x):
        """Stable softmax implementation"""
        try:
            n = x.shape[1]
        except IndexError:
            n = 1
        out_dim = x.shape[0]
        max_x = np.max(x, axis=0).reshape(1, n) if not n == 1 else np.max(x)
        z = x - max_x
        numerator = exp(z)
        denominator = np.sum(numerator, axis=0).reshape(1, n) if not out_dim == 1 else sum(numerator)
        return numerator / denominator

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)
