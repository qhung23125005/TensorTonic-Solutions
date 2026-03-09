import numpy as np

def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    # Write code here
    series = np.asarray(series)
    n = len(series)
    
    mean = np.mean(series)
    gamma0 = np.sum((series - mean)**2)

    if (gamma0 == 0):
        result = [0]*(max_lag+1)
        result[0] = 1
        return result

    r = np.zeros(max_lag + 1)

    for k in range(max_lag + 1):
        r[k] = np.sum((series[:(n-k)] - mean)*(series[k:] - mean))/gamma0 
    return r.tolist()
    