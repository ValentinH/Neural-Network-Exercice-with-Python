
import scipy.io
import numpy as np

from display_data import display_data
from nn_cost_function import nn_cost_function
from rand_initialize_weights import rand_initialize_weights
from sigmoid import sigmoid_gradient

input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10

mat = scipy.io.loadmat('data/numbers.mat')
X = np.array(mat['X'])
y = np.array(mat['y'])

weights = scipy.io.loadmat('data/weights.mat')
# decrease all example value by 1 since python is 0-based (0 will be K=1,  9->K=0)
# for my tests, I will do 0->k=0
y_test = [y_val - 1 for y_val in y]

m = X.shape[0]

random_indices = np.random.randint(m, size=100)
# display_data(X[random_indices])
print('Display selected inputs')
#input()

print('Tests nn_cost_function with default data')

test_theta1 = weights['Theta1']
test_theta2 = weights['Theta2']

nn_params = np.concatenate((test_theta1.ravel(), test_theta2.ravel()))

_lambda = 0
j = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y_test, _lambda)
print('\nCost without reg', j)
print('(this value should be about 0.287629) \n')

_lambda = 1
j = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y_test, _lambda)
print('\nReal cost', j)
print('(this value should be about 0.383770) \n')


print('Initializing Neural Network Parameters for real computation...')
#  real computation

#initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
#initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
# Unroll parameters
#initial_nn_params = np.concatenate((initial_Theta1.T.ravel(), initial_Theta2.T.ravel()))