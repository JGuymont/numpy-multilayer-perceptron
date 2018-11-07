import numpy as np

from mlp.mlp import MLPClassifier

CIRCLE_DATA_PATH = './data/circles/circles.txt'

if __name__ == '__main__':
    data = np.loadtxt(open(CIRCLE_DATA_PATH, 'r'))
    
    X = data[:, 0:2]
    y = data[:, -1]

    X_dim = X.shape[1]
    y_dim = 2 # 0 or 1
    
    # hyperparameters
    HIDDEN_DIM = 2

    mlp = MLPClassifier(X_dim, HIDDEN_DIM, y_dim)
    



    