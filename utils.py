import numpy as np


# noinspection PyTypeChecker
def rmspe(y, y_hat):
    return np.sqrt(np.mean((y_hat / y - 1) ** 2))