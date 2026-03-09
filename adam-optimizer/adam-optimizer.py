import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param = np.asarray(param)
    grad = np.asarray(grad)
    m = np.asarray(m)
    v = np.asarray(v)
    t = np.asarray(t)
    
    # Update biased first moment estimate
    m_new = beta1 * m + (1 - beta1) * grad

    # Update biased second raw moment estimate
    v_new = beta2 * v + (1 - beta2) * (grad * grad)

    # Bias correction
    m_hat = m_new / (1 - np.pow(beta1,t))
    v_hat = v_new / (1 - np.pow(beta2, t))

    # Parameter update
    param_new = param - lr*m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_new, v_new
    