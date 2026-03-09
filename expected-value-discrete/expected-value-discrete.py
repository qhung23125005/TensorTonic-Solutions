import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x)
    p = np.asarray(p)
    if abs(np.sum(p) - 1) > 1e-6:
        raise ValueError
    return np.sum(x*p)
