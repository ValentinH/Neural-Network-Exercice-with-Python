from nn_functions import compute_cost, compute_gradients, recode_labels
from toolbox import debug_initialize_weights
import numpy as np


def check_nn_gradients(_lambda=0):

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    x = debug_initialize_weights(m, input_layer_size - 1)
    y = np.mod(np.arange(1, m+1), num_labels).T

    yk = recode_labels(y, num_labels)
    x_bias = np.r_[np.ones((1, x.shape[0] )), x.T]

    nn_params = np.concatenate((theta1.T.ravel(), theta2.T.ravel()))

    def cost_function(p):
        return compute_cost(p, input_layer_size, hidden_layer_size, num_labels, x, y, _lambda, yk, x_bias)

    def gradients_function(p):
        return compute_gradients(p, input_layer_size, hidden_layer_size, num_labels, x, y, _lambda, yk, x_bias)

    gradients = gradients_function(nn_params)
    num_gradients = compute_numerical_gradient(cost_function, nn_params)

    for i in range(len(gradients)):
        print(num_gradients[i], gradients[i])

    print('The above two columns you get should be very similar.\n',
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

    diff = np.linalg.norm(num_gradients-gradients) / np.linalg.norm(num_gradients+gradients)

    print('If your backpropagation implementation is correct, then \n',
          'the relative difference will be small (less than 1e-9). \n',
          '\nRelative Difference: ', diff)


def compute_numerical_gradient(cost_fn, theta):

    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = e
        loss1 = cost_fn(theta - perturb)
        loss2 = cost_fn(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    return numgrad
