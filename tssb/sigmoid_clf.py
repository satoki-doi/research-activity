# -*- coding: utf-8 -*-

import math
import numpy as np

def sigmoid_clf(z):
    z = 1 / (1 + math.exp(-z))
    if z >= 0.5:
        return 1.0
    else:
        return 0.0


def feature_trans(array):
    output = np.array([])
    for arr in array:
        output = np.append(output, map(sigmoid_clf, arr))
    return output.reshape(array.shape)
