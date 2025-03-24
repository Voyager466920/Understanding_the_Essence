import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(matrix, title, xlabel="Columns", ylabel="Rows"):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)


def one_hot(vector, size):
    vec = np.zeros(size)
    vec[vector] = 1
    return vec