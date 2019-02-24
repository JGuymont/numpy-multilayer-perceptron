import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import time

from constants import MNIST as constants
from utils.dataset import Dataset
from utils.dataloader import DataLoader
from utils import visualize
from neural_network.mlp import MLPClassifier


if __name__ == '__main__':

    train_data = Dataset(
        path=constants.TRAIN_PATH,
        input_features=constants.INPUT_FEATURES,
        output_features=constants.OUTPUT_FEATURES,
        header=0,
        transform=lambda X: [x / 255 for x in X]
    )

    valid_data = Dataset(
        path=constants.VALID_PATH,
        input_features=constants.INPUT_FEATURES,
        output_features=constants.OUTPUT_FEATURES,
        header=0,
        transform=lambda X: [x / 255 for x in X]
    )

    trainloader = DataLoader(train_data, batch_size=constants.BATCH_SIZE)
    devloader = DataLoader(valid_data, batch_size=1000)

    mlp = MLPClassifier(
        input_size=constants.INPUT_DIM,
        hidden_size=constants.HIDDEN_DIM,
        output_size=constants.N_CLASSES,
        learning_rate=constants.LEARNING_RATE,
        num_epochs=constants.NUM_EPOCHS
    )

    loss_storage, acc_storage = mlp.train(
        trainloader,
        devloader,
        log=os.path.join(constants.RESULTS_DIR, 'mnist_log.txt')
    )

    visualize.plot_mnist_results(
        loss_storage,
        acc_storage,
        out_path=os.path.join(constants.FIGURES_DIR, 'mnist_results.png')
    )
