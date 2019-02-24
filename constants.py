import os


class Circles:
    DATA_DIR = './data/circles/'
    DATA_PATH = os.path.join(DATA_DIR, 'circles.txt')
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    VALID_PATH = os.path.join(DATA_DIR, 'valid.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
    RESULTS_DIR = './results/circles/'
    FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures/')

    INPUT_FEATURES = [0, 2]
    OUTPUT_FEATURES = 2
    INPUT_DIM = 2
    N_CLASSES = 2


class MNIST:
    DATA_DIR = './data/mnist/'
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    VALID_PATH = os.path.join(DATA_DIR, 'valid.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
    RESULTS_DIR = './results/mnist/'
    FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures/')

    INPUT_FEATURES = [1, 785]
    OUTPUT_FEATURES = 0
    INPUT_DIM = 784
    N_CLASSES = 10

    # hyperparameters
    HIDDEN_DIM = 50
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.01
