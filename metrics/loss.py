import numpy as np
from metrics.function_speed import measure_speed


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true-y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_loss(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

def binary_cross_entropy(y_true, y_pred):
     epsilon = 1e-7
     y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
     m =  y_pred.shape[0]
     return -1/m * (np.dot(y_true, np.log(y_pred)) + np.dot((1-y_true), np.log(1-y_pred)))


