import numpy as np
import sys
from scipy.optimize import fmin_cg

from toolbox import *


def feed_forward(theta1, theta2, x, x_bias=None):
    one_rows = np.ones((1, x.shape[0]))

    a1 = np.r_[one_rows, x.T] if x_bias is None else x_bias
    z2 = theta1.dot(a1)
    a2 = sigmoid(z2)
    a2 = np.r_[one_rows, a2]
    z3 = theta2.dot(a2)
    a3 = sigmoid(z3)

    return a1, a2, a3, z2, z3


def compute_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, _lambda, yk=None, x_bias=None):
    m = x.shape[0]
    theta1, theta2 = unroll_thetas(nn_params, input_layer_size, hidden_layer_size, num_labels)
    a1, a2, a3, z2, z3 = feed_forward(theta1, theta2, x, x_bias)

    if yk is None:
        yk = recode_labels(y, num_labels)
        assert yk.shape == a3.shape, 'Error, shape of recoded y is different from a3'

    term1 = -yk * np.log(a3)
    term2 = (1 - yk) * np.log(1 - a3)
    left_term = np.sum(term1 - term2) / m
    right_term = np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2)

    return left_term + right_term * _lambda / (2 * m)


def compute_gradients(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, _lambda, yk=None, x_bias=None):
    m = x.shape[0]
    theta1, theta2 = unroll_thetas(nn_params, input_layer_size, hidden_layer_size, num_labels)
    a1, a2, a3, z2, z3 = feed_forward(theta1, theta2, x, x_bias)

    if yk is None:
        yk = recode_labels(y, num_labels)
        assert yk.shape == a3.shape, 'Error, shape of recoded y is different from a3'

    # Backward propagation to compute gradients
    sigma3 = a3 - yk
    sigma2 = theta2.T.dot(sigma3) * sigmoid_gradient(np.r_[np.ones((1, m)), z2])
    sigma2 = sigma2[1:, :]

    accum1 = sigma2.dot(a1.T) / m
    accum2 = sigma3.dot(a2.T) / m
    accum1[:, 1:] = accum1[:, 1:] + (theta1[:, 1:] * _lambda / m)
    accum2[:, 1:] = accum2[:, 1:] + (theta2[:, 1:] * _lambda / m)
    accum = np.array([accum1.T.reshape(-1).tolist() + accum2.T.reshape(-1).tolist()]).T

    return np.ndarray.flatten(accum)


def predict(theta1, theta2, x):
    m = x.shape[0]
    a1, a2, a3, z2, z3 = feed_forward(theta1, theta2, x)
    return np.reshape(np.argmax(a3, axis=0), (m, 1))


iterations_counter = 0


def train_model(x, y, input_layer_size, hidden_layer_size, num_labels):

    initial_theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    initial_nn_params = np.concatenate((initial_theta1.T.ravel(), initial_theta2.T.ravel()))

    _lambda = 0.1
    max_iterations = 100
    iterations_counter = 0

    yk = recode_labels(y, num_labels)
    x_bias = np.r_[np.ones((1, x.shape[0])), x.T]

    def show_progress(current_x):
        global iterations_counter
        iterations_counter += 1
        progress = iterations_counter * 100 // max_iterations
        sys.stdout.write('\r[{0}{1}] {2}% - iter:{3}'.format(
            '=' * (progress // 5),
            ' ' * ((104 - progress) // 5),
            progress, iterations_counter
        ))

    # Solve!
    nn_params = fmin_cg(
        compute_cost,
        x0=initial_nn_params,
        args=(input_layer_size, hidden_layer_size, num_labels, x, y, _lambda, yk, x_bias),
        fprime=compute_gradients,
        maxiter=max_iterations,
        callback=show_progress
    )

    # Obtain theta1 and theta2 back from nn_params
    theta1, theta2 = unroll_thetas(nn_params, input_layer_size, hidden_layer_size, num_labels)

    return theta1, theta2
