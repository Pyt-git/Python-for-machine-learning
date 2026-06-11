import numpy as np

# ---------------------------
# Input validation
# ---------------------------

def safe_forward(tokens, params, P):
    # 1) VALIDATION
    if not np.issubdtype(tokens.dtype, np.integer):
        raise TypeError("Tokens must be integers")

    if np.any(tokens < 0):
        raise ValueError("Tokens must be non‑negative")

    vocab_size = params["embed"].shape[0]
    if np.any(tokens >= vocab_size):
        raise ValueError("Token out of vocabulary range")

    # 2) TRANSFORMER FORWARD
    return transformer_forward(tokens, params, P)

# -----------------------------
# Utility
# -----------------------------

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# -----------------------------
# Positional embeddings
# -----------------------------

def build_sinusoidal_positional_embeddings(T_max, d_model):
    P = np.zeros((T_max, d_model))
    positions = np.arange(T_max)[:, None]          # (T_max, 1)
    dims = np.arange(d_model)[None, :]             # (1, d_model)
    angles = positions / (10000 ** (dims / d_model))
    P[:, 0::2] = np.sin(angles[:, 0::2])
    P[:, 1::2] = np.cos(angles[:, 1::2])
    return P                                      # (T_max, d_model)

def add_positional_embeddings(X, P):
    """
    X: (B, T, d_model)
    P: (T_max, d_model)
    """
    B, T, d_model = X.shape
    P_slice = P[:T]                               # (T, d_model)
    return X + P_slice[None, :, :]                # (B, T, d_model)

# -----------------------------
# Causal mask
# -----------------------------

def build_causal_mask(T):
    mask = np.triu(np.ones((T, T)), k=1)          # (T, T), 1 above diag
    mask = mask * -1e9
    return mask[None, None, :, :]                 # (1, 1, T, T)

# -----------------------------
# LayerNorm
# -----------------------------

def layernorm(X, gamma, beta, eps=1e-5):
    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    X_norm = (X - mean) / np.sqrt(var + eps)
    return X_norm * gamma + beta

# -----------------------------
# Parameter initialization
# -----------------------------

def init_transformer_block_params(d_model, h, d_ff):
    d_k = d_model // h
    params = {}

    # Attention projections
    params["W_q"] = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    params["W_k"] = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    params["W_v"] = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    params["W_o"] = np.random.randn(d_model, d_model) / np.sqrt(d_model)

    # FFN
    params["W1"] = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
    params["W2"] = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)

    # LayerNorms
    params["ln1_gamma"] = np.ones(d_model)
    params["ln1_beta"]  = np.zeros(d_model)
    params["ln2_gamma"] = np.ones(d_model)
    params["ln2_beta"]  = np.zeros(d_model)

    # Meta
    params["h"] = h
    params["d_model"] = d_model
    params["d_k"] = d_k

    return params

def init_transformer_params(vocab_size, d_model, h, d_ff, T_max):
    params = {}
    params["embed"] = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)
    params["block"] = init_transformer_block_params(d_model, h, d_ff)
    params["proj"]  = np.random.randn(d_model, vocab_size) / np.sqrt(d_model)
    params["T_max"] = T_max
    params["d_model"] = d_model
    return params

# -----------------------------
# Multi‑Head Attention
# -----------------------------

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

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores, axis=-1)                   # (B, h, T, T)
    heads_out = weights @ V                              # (B, h, T, d_k)

    combined = heads_out.transpose(0, 2, 1, 3).reshape(B, T, d_model)
    return combined @ W_o                                # (B, T, d_model)

# -----------------------------
# Feed‑Forward Network
# -----------------------------

def ffn(X, params):
    W1 = params["W1"]
    W2 = params["W2"]
    H = np.maximum(0, X @ W1)
    return H @ W2

# -----------------------------
# Dropout (simple, training‑only)
# -----------------------------

def dropout(X, p):
    if p <= 0.0:
        return X
    mask = (np.random.rand(*X.shape) > p).astype(X.dtype)
    return X * mask / (1.0 - p)

# -----------------------------
# Single Transformer block
# -----------------------------

def transformer_block(X, block_params, dropout_p=0.1):
    B, T, d_model = X.shape
    mask = build_causal_mask(T)

    # Attention
    A = multihead_attention(X, block_params, mask=mask)
    A = dropout(A, dropout_p)
    X = X + A
    X = layernorm(X, block_params["ln1_gamma"], block_params["ln1_beta"])

    # FFN
    F = ffn(X, block_params)
    F = dropout(F, dropout_p)
    X = X + F
    X = layernorm(X, block_params["ln2_gamma"], block_params["ln2_beta"])

    return X

# -----------------------------
# Full Transformer forward
# -----------------------------

def transformer_forward(tokens, params, P, dropout_p=0.1):
    """
    tokens: (B, T) int
    P: (T_max, d_model)
    """
    B, T = tokens.shape
    d_model = params["d_model"]

    embed = params["embed"]                     # (V, d_model)
    X = embed[tokens]                           # (B, T, d_model)

    X = add_positional_embeddings(X, P)         # (B, T, d_model)
    X = transformer_block(X, params["block"], dropout_p=dropout_p)
    logits = X @ params["proj"]                 # (B, T, V)
    return logits

# -----------------------------
# Cross‑entropy loss
# -----------------------------

def cross_entropy_loss(logits, targets):
    """
    logits: (B, T, V)
    targets: (B, T)
    """
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)

    probs = softmax(logits_flat, axis=-1)
    correct = probs[np.arange(B * T), targets_flat]
    loss = -np.log(correct + 1e-9).mean()
    return loss
