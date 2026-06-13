import numpy as np

def dropout(X, p):
    if p <= 0.0:
        return X
    mask = (np.random.rand(*X.shape) > p).astype(X.dtype)
    return X * mask / (1.0 - p)

