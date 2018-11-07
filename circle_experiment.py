import numpy as np

from mlp.mlp import MLPClassifier

CIRCLE_DATA_PATH = './data/circles/circles.txt'

# hyperparameters
HIDDEN_DIM = 10
BATCH_SIZE = 1
NUM_EPOCHS = 100000
LEARNING_RATE = 0.01

if __name__ == '__main__':
    data = np.loadtxt(open(CIRCLE_DATA_PATH, 'r'))
    
    X = data[:, 0:2]
    y = data[:, -1].astype(int)

    X_dim = X.shape[1]
    y_dim = 2 # 0 or 1

    

    mlp = MLPClassifier(X_dim, HIDDEN_DIM, y_dim, LEARNING_RATE, NUM_EPOCHS)

    #y = np.array([0, 1])
    #onehot = mlp._onehot(y)
    #print(y)
    #print(onehot)

    acc = mlp.accuracy(X,y)
    print(acc)
    mlp.train(zip(X, y))
    acc = mlp.accuracy(X,y)
    print(acc)



    