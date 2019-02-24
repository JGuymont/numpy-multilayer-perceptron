import matplotlib.pyplot as plt
import numpy as np


def plot_decision(X, y, path, model, param, ax=None, h=0.07):
    """
    plot the decision boundary. `h` controls plot quality.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    # https://stackoverflow.com/a/19055059/6027071
    # sample a region larger than our training data X
    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # plot decision boundaries
    x = np.concatenate(([xx.ravel()], [yy.ravel()]))
    pred = model.predict(x.T).reshape(xx.shape)
    ax.contourf(xx, yy, pred, alpha=0.8, cmap='RdYlBu')

    # plot points (coloured by class)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, cmap='RdYlBu')
    ax.axis('off')

    title = 'hidden_dim: {} | learning rate: {} | n_epochs: {} | lambda_1: {} | lambda_2: {}'.format(
        param[0], param[1], param[2], param[3], param[4]
    )

    plt.title(title)
    plt.savefig(path)
    plt.close()


def plot_gradient(grad1, grad2, param_names, legend, path):
    plt.rcParams.update({'font.size': 6})
    plt.plot(param_names, grad1, '--')
    plt.plot(param_names, grad2, 'o')
    plt.legend(legend)
    plt.xlabel('parameter')
    plt.ylabel('gradient')
    plt.savefig(path)


def plot_result(X_train, Y_train, X_test, Y_test, predictions, title=None, y_lim=None):
    _, ax = plt.subplots()
    if y_lim:
        ax.set_ylim(y_lim)
    ax.plot(X_train, Y_train, "o", label='Training set D_n')
    ax.plot(X_test, Y_test, '-', label='h(x)')

    for label, pred in predictions.items():
        ax.plot(X_test, pred, label=label)

    ax.legend()
    plt.savefig('{}.png'.format(title), bbox_inches='tight')
    plt.close()


def plot_mnist_results(loss_storage, acc_storage, out_path):
    i = 0
    plt.figure(1)
    plot_id = [221, 222, 223, 224]
    for dataset, losses in loss_storage.items():
        plt.subplot(plot_id[i])
        plt.plot(losses)
        plt.xlabel('epoch')
        plt.ylabel('Loss {}'.format(dataset))
        i += 1

    for dataset, acc in acc_storage.items():
        plt.subplot(plot_id[i])
        plt.plot(acc)
        plt.xlabel('epoch')
        plt.ylabel('Acc. {}'.format(dataset))
        i += 1

    plt.savefig(out_path)
    plt.close()
