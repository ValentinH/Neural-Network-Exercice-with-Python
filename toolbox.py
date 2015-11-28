import math
import numpy as np


def sigmoid(z):
    return np.true_divide(1.0, 1.0 + np.exp(-z))


def sigmoid_gradient(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


def debug_initialize_weights(fan_out, fan_in):
    length = fan_out * (fan_in + 1)

    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    return np.reshape(np.sin(np.arange(1, length+1)), (fan_out, fan_in + 1), order='F') / 10


def rand_initialize_weights(l_in, l_out):

    epsilon_init = math.sqrt(6)/math.sqrt(l_in + l_out)
    return np.random.rand(l_out, l_in+1) * 2 * epsilon_init - epsilon_init


def unroll_thetas(nn_params, input_layer_size, hidden_layer_size, num_labels):
    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)), order='F')
    theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)), order='F')

    return theta1, theta2


def recode_labels(y, num_labels):
    # convert y to use one vector for each value
    return np.identity(num_labels)[y].squeeze().T