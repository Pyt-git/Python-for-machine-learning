import numpy as np

def transformer_block(B, T, d_model, h, d_ff, dropout_p=0.1):
    d_k = d_model // h

    # -----------------------------
    # 1) Dropout
    # -----------------------------
    def dropout(X, p=dropout_p):
        mask = (np.random.rand(*X.shape) > p).astype(X.dtype)
        return X * mask / (1 - p)

    # -----------------------------
    # 2) LayerNorm
    # -----------------------------
    def layernorm(X, eps=1e-5):
        mean = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)
        X_norm = (X - mean) / np.sqrt(var + eps)
        gamma = np.ones(d_model)
        beta = np.zeros(d_model)
        return X_norm * gamma + beta

    # -----------------------------
    # 3) Multi‑Head Attention
    # -----------------------------
    def multihead_attention(X):
        # Q, K, V projections
        W_q = np.random.randn(d_model, d_model)
        W_k = np.random.randn(d_model, d_model)
        W_v = np.random.randn(d_model, d_model)

        Q = X @ W_q
        K = X @ W_k
        V = X @ W_v

        # reshape into heads
        Q = Q.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)

        # scaled dot‑product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)

        def softmax(x):
            x = x - np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(x)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        weights = softmax(scores)
        heads_out = weights @ V

        # combine heads
        combined = heads_out.transpose(0, 2, 1, 3).reshape(B, T, d_model)

        # final projection
        W_o = np.random.randn(d_model, d_model)
        return combined @ W_o

    # -----------------------------
    # 4) Feed‑Forward Network
    # -----------------------------
    def ffn(X):
        W1 = np.random.randn(d_model, d_ff)
        W2 = np.random.randn(d_ff, d_model)
        H = np.maximum(0, X @ W1)
        return H @ W2

    # -----------------------------
    # 5) Full block logic
    # -----------------------------
    X = np.random.randn(B, T, d_model)

    # Multi‑head attention + residual + dropout + LN
    A = multihead_attention(X)
    A = dropout(A)
    X = layernorm(X + A)

    # FFN + residual + dropout + LN
    F = ffn(X)
    F = dropout(F)
    X = layernorm(X + F)

    return X
