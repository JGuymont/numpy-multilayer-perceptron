import numpy as np

CIRCLE_DATA_PATH = './data/circles.txt'

if __name__ == '__main__':
    data = np.loadtxt(open(CIRCLE_DATA_PATH, 'r'))
    
    x = data[0]
    print(x)

    