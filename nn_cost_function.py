import numpy as np

from sigmoid import sigmoid, sigmoid_gradient


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, _lambda):

    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))
    theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    m = x.shape[0]

    y = np.identity(num_labels)[y].squeeze()  # convert y to use one vector for each value

    # Forward propagation to compute a3
    a1 = np.c_[np.ones((m, 1)), x].T  # add a column with 1 to X for bias
    z2 = np.dot(theta1, a1)
    a2 = sigmoid(z2)
    a2 = np.r_[np.ones((1, a2.shape[1])), a2]
    z3 = np.dot(theta2, a2)
    a3 = sigmoid(z3)

    # compute the cost without regularization
    j = 0
    for k in range(num_labels):
        costs = -y[:, k].T * np.log(a3[k, :]) - (1 - y[:, k].T) * np.log(1 - a3[k, :])
        j += 1/m * np.sum(costs)

    # add regularization
    reg = _lambda/(2*m) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    j += reg

    # Backward propagation to compute gradients
    delta3 = (a3 - y.T)
    delta2 = np.dot(theta2[:, 1:].T, delta3) * sigmoid_gradient(z2)
    delta1 = np.dot(delta2, a1.T)
    delta2 = np.dot(delta3, a2.T)

    theta1_grad = (1/m) * delta1
    theta2_grad = (1/m) * delta2

    # Regularize gradients
    temp1 = theta1
    temp1[:, 0] = 0
    theta1_grad += np.dot((_lambda/m),  temp1)

    temp2 = theta2
    temp2[:, 0] = 0
    theta2_grad += np.dot((_lambda/m), temp2)

    gradients = np.concatenate((theta1_grad.ravel(), theta2_grad.ravel()))

    return j, gradients

