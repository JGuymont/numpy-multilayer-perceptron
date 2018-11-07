import numpy as np
from numpy import exp

class Activation:

    @staticmethod
    def softmax(x):
        """Stable softmax implementation"""
        try:
            d = x.shape[1]
        except IndexError:
            d = 1
        n = x.shape[0]
        max_x = np.max(x, axis=1).reshape(n, 1) if not d == 1 else np.max(x) 
        z = x - max_x
        numerator = exp(z)
        denominator = np.sum(numerator, axis=1).reshape(n, 1) if not d == 1 else sum(numerator)
        return numerator / denominator

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)


if __name__ == '__main__':
    activation = Activation()
    x = np.array([[-1, 2, -3, 4],[-5, 6, 7, -1]])
    softmax_ = activation.softmax(x)
    relu_ = activation.relu(x)

    print('unit test softmax:\n', softmax_)
    print('unit test relu:\n', relu_)