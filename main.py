
import scipy.io
import numpy as np
import sys
from scipy.optimize import fmin_cg

from display_data import display_data
from nn_functions import compute_cost, predict, train_model
from toolbox import *
from check_nn_gradients import check_nn_gradients

input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10

mat = scipy.io.loadmat('data/numbers.mat')
X = np.array(mat['X'])
y = np.array(mat['y'])

weights = scipy.io.loadmat('data/weights.mat')
# decrease all example value by 1 since python is 0-based (0 will be K=1,  9->K=0)
# for my tests, I will do 0->k=0
y_test = np.array([y_val - 1 for y_val in y])

m = X.shape[0]

random_indices = np.random.randint(m, size=100)
#display_data(X[random_indices])
print('Display selected inputs')
#input()

print('Tests nn_cost_function with default data')

test_theta1 = weights['Theta1']
test_theta2 = weights['Theta2']

nn_params = np.concatenate((test_theta1.T.ravel(), test_theta2.T.ravel()))

_lambda = 0
j = compute_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y_test, _lambda)
print('\nCost without reg', j)
print('(this value should be about 0.287629) \n')

_lambda = 1
j = compute_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y_test, _lambda)
print('\nReal cost', j)
print('(this value should be about 0.383770) \n')

print('\nEvaluating sigmoid gradient...\n')

g = sigmoid_gradient(np.array([1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ')
print(g)
print('\n\n')

print('Check gradients')
check_nn_gradients()

print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
_lambda = 3
check_nn_gradients(_lambda)

# Also output the costFunction debugging values
debug_J = compute_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y_test, _lambda)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = 3): ', debug_J,
      '\n(this value should be about 0.576051)')

print('\nPress enter to start training:')
input()
print('\nTraining Neural Network... \n')

theta1, theta2 = train_model(X, y_test, input_layer_size, hidden_layer_size, num_labels)

print('\nVisualizing Neural Network... \n')
#display_data(theta1[:, 1:])

# predict using trained parameters
pred = predict(theta1, theta2, X)
print('\nTraining Set Accuracy: ', np.mean(pred == y_test) * 100)
