import numpy as np
from sigmoid import sigmoid


def predict(theta1, theta2, x):

    # PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(theta1, theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (theta1, theta2)

    # Useful values
    m = x.shape[0]

    a1 = np.c_[np.ones((m, 1)), x].T  # add a column with 1 to X for bias
    z2 = np.dot(theta1, a1)
    a2 = sigmoid(z2)
    a2 = np.r_[np.ones((1, a2.shape[1])), a2]
    z3 = np.dot(theta2, a2)
    a3 = sigmoid(z3)

    return np.reshape(np.argmax(a3, axis=0), (m, 1))
