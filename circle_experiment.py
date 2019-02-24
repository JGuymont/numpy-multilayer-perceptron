import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import random
import time

from utils.dataset import Dataset
from utils.dataloader import DataLoader
from utils import visualize
from neural_network.mlp import MLPClassifier
import constants


def finite_difference_check(dataset, batch_size):
    """
    Computes the gradients for a single example, and
    check that the gradient is correct using the nite
    difference method.

    Answers to questions 1, 2, and 4.
    """
    dataloader = DataLoader(dataset, batch_size)

    inputs, targets = next(dataloader)

    mlp = MLPClassifier(constants.Circles.INPUT_DIM, constants.Circles.N_CLASSES)
    gradHats, grads, param_names = mlp.finite_difference_check(inputs, targets)

    figure_name = 'finite_difference_check_batch_size_{}.png'.format(batch_size)

    visualize.plot_gradient(
        gradHats, grads,
        param_names,
        legend=['finite differences approx.', 'backpropagation'],
        path=os.path.join(constants.Circles.FIGURES_DIR, figure_name)
    )


def decision_boundaries(train_data, valid_data):
    """
    Question 5: Train the neural network using gradient descent on the two circles dataset.
    Plot the decision regions for several different values of the hyperparameters
    (weight decay, number of hidden units, early stopping) so as to illustrate their
    effect on the capacity of the model.
    """

    # raw data is only used to plot the decision boundary
    raw_data = np.loadtxt(open(constants.Circles.DATA_PATH, 'r'))
    X = raw_data[:, :2]
    y = raw_data[:, -1]

    # hyperparameters
    HIDDEN_DIM_SET = [8, 14]
    NUM_EPOCH_SET = [30]
    LEARNING_RATE_SET = [0.05]
    L1_WEIGH_DECAY = [0, 0.005]
    L2_WEIGH_DECAY = [0, 0.005]

    trainloader = DataLoader(train_data, batch_size=32)
    devloader = DataLoader(valid_data, batch_size=len(valid_data))

    i = 0
    for h in HIDDEN_DIM_SET:
        for lr in LEARNING_RATE_SET:
            for l1 in L1_WEIGH_DECAY:
                for l2 in L2_WEIGH_DECAY:
                    for n_epoch in NUM_EPOCH_SET:

                        print('\nhidden_dim: {}, lr: {}, l1: {}, l2: {}'.format(h, lr, l1, l2))
                        mlp = MLPClassifier(constants.Circles.INPUT_DIM, constants.Circles.N_CLASSES, h, lr, n_epoch, l1, l2, l1, l2)
                        mlp.train(trainloader, devloader)

                        figure_name = 'decision_boundaries_{}.png'.format(i)

                        visualize.plot_decision(
                            X, y,
                            path=os.path.join(constants.Circles.FIGURES_DIR, figure_name),
                            model=mlp,
                            param=[h, lr, n_epoch, l1, l2, l1, l2]
                        )
                        i += 1


def main_experiment(train_data, valid_data):
    """
    Question 8:
    """
    trainloader = DataLoader(train_data, batch_size=32)
    devloader = DataLoader(valid_data, batch_size=len(valid_data))

    mlp = MLPClassifier(constants.Circles.INPUT_DIM, constants.Circles.N_CLASSES, 10, 0.05, 50)
    mlp.train(trainloader, devloader, log=os.path.join(constants.Circles.RESULTS_DIR, 'circles_log.txt'))


if __name__ == '__main__':

    train_data = Dataset(
        path=constants.Circles.TRAIN_PATH,
        input_features=constants.Circles.INPUT_FEATURES,
        output_features=constants.Circles.OUTPUT_FEATURES
    )

    valid_data = Dataset(
        path=constants.Circles.VALID_PATH,
        input_features=constants.Circles.INPUT_FEATURES,
        output_features=constants.Circles.OUTPUT_FEATURES
    )

    finite_difference_check(train_data, batch_size=1)
    finite_difference_check(train_data, batch_size=10)
    decision_boundaries(train_data, valid_data)
    main_experiment(train_data, valid_data)
