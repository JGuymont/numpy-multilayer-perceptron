import numpy as np
from numpy import exp

def softmax(x):
    max_x = np.max(x, axis=1).reshape(x.shape[0], 1)
    z = x - max_x
    numerator = exp(z)
    denominator = np.sum(exp(z), axis=1).reshape(x.shape[0], 1)
    return numerator / denominator

if __name__ == '__main__':
    x = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])
    softmax_ = softmax(x)
    print(softmax_) 