import math
import numpy as np


def rand_initialize_weights(l_in, l_out):

    epsilon_init = math.sqrt(6)/math.sqrt(l_in + l_out)
    return np.random.rand(l_out, l_in+1) * 2 * epsilon_init - epsilon_init
