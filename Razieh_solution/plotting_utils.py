import matplotlib.pyplot as plt
import numpy as np

def plot_train_acc(train_acc, test_acc, epochs, filename, ylabel):
    fig = plt.figure(facecolor='w', figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(epochs, train_acc, label="train")
    ax.plot(epochs, test_acc, label="test")
    ax.set_xlabel('Epochs', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    legend = ax.legend(fontsize=15)
    plt.savefig(filename)