
import numpy as np


def sigmoid(z):
    return np.true_divide(1.0, 1.0 + np.exp(-z))


def sigmoid_gradient(z):
    return np.dot(sigmoid(z), 1.0 - sigmoid(z))
