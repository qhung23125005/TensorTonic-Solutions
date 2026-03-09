import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    s = 1/(1 + np.exp(-np.asarray(x, dtype=float)))
    return s