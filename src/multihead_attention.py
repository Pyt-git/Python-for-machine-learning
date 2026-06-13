import numpy as np

def multihead_attention(X, params, mask=None):
    B, T, d_model = X.shape
    h   = params["h"]
    d_k = params["d_k"]

    W_q = params["W_q"]
    W_k = params["W_k"]
    W_v = params["W_v"]
    W_o = params["W_o"]

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    Q = Q.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)  # (B, h, T, d_k)
    K = K.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)

    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)  # (B, h, T, T)

