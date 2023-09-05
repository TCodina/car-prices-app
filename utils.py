import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_dataset(x, y, features_list):
    N, D = x.shape

    fig, axes = plt.subplots(4, D // 3, figsize=(15, 20))
    colors = list(mcolors.BASE_COLORS)
    for i in range(D):
        ax = axes.flatten()[i]
        ax.plot(x[:, i], y, 'o', color=colors[i], alpha=0.5)
        ax.grid()
        ax.set_xlabel(f"{features_list[i]} ($x_{i}$)")
        ax.set_ylabel(f"Price ($y$)")
    plt.suptitle('Prices vs Features')
    plt.show()


def plot_history(history_dic):
    loss_train = history_dic['train']
    loss_test = history_dic['test']
    parameters = history_dic['weights'].squeeze()

    best_epoch = np.argmin(loss_test)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8))

    ax0.plot(loss_train, 'b-', label='Train')
    ax0.plot(loss_test, 'r-', label='Test')
    ax0.axvline(best_epoch, color='k', label='Best epoch')
    ax0.set_xlabel('Epochs')
    ax0.set_ylabel('Loss')
    ax0.set_title('Evolution of training and test loss')
    ax0.legend()
    ax0.grid()

    for param in parameters.T:
        ax1.plot(param)
    ax1.axvline(best_epoch, color='k')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Parameters')
    ax1.set_title('Evolution of parameters')
    ax1.grid()

    plt.show()


def plot_with_predicted(x, y, y_pred, features_list):
    N, D = x.shape

    fig, axes = plt.subplots(4, D // 3, figsize=(15, 20))
    for i in range(D):
        ax = axes.flatten()[i]
        ax.plot(x[:, i], y, 'ro', alpha=0.5, label='ground truth')
        ax.plot(x[:, i], y_pred, 'bo', alpha=0.5, label='predicted')
        ax.set_xlabel(f"{features_list[i]} ($x_{i}$)")
        ax.set_ylabel(f"Price ($y$)")
        ax.legend()
        ax.grid()
    plt.suptitle('Prices vs Features')
    plt.show()