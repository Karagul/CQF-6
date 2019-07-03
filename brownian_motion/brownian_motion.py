# From Module 1 Lecture 3
# Brownian motion example with decreasing time steps.
import numpy as np
import math
import matplotlib.pyplot as plt


def main():
    """

    Main controlling function. Set parameters. Array [n] contains the number of steps.
    :return:
    """
    n = np.array([10, 25, 1000])
    x_values, paths = bm(n)
    plot_paths(x_values, paths)


def bm(n: np.ndarray):
    """

    Create two lists - one with the x-values, one with the accumulated y-values.
    :param n: Array with the number of steps.
    :return: Two lists of numpy arrays.
    """
    paths = []
    x_values = []

    for i in np.nditer(n):
        x, p = payoff(i)
        paths.append(p)
        x_values.append(x)
    return x_values, paths


def payoff(n):
    """

    Generate two arrays - one with the x-values, one with the accumulated y-values.
    :param n: Numpy array with integer for number of time steps.
    :return:
    """
    pay = np.zeros([n])
    i = 0
    while i < n:
        if np.random.uniform(0, 1) > 0.5:
            p = 1
        else:
            p = -1
        if i == 0:
            pay[i] = 0
        else:
            pay[i] = pay[i-1] + p * math.sqrt(1/n)
        i += 1
    x = np.linspace(0, 1, num=n)
    return x, pay


def plot_paths(x: list, l: list):
    """

    Plot all arrays.
    :param x: List of arrays of x-values.
    :param l: List of arrays of y-values.
    :return:
    """
    i = 0
    while i < len(l):
        plt.plot(x[i], l[i])
        i += 1
    plt.show()


if __name__ == '__main__':
    main()
