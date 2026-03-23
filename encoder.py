import numpy as np
from attention import scaled_dot_product_attention

D_MODEL = 64
D_FF = 256
EPS = 1e-6


def layer_norm(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS)


def ffn(x, W1, W2):
    return np.maximum(0, x @ W1) @ W2


def run_encoder(input_ids, embedding_table):
    X = embedding_table[input_ids][np.newaxis, :, :]

    WQ = np.random.randn(D_MODEL, D_MODEL)
    WK = np.random.randn(D_MODEL, D_MODEL)
    WV = np.random.randn(D_MODEL, D_MODEL)

    W1 = np.random.randn(D_MODEL, D_FF)
    W2 = np.random.randn(D_FF, D_MODEL)

    attn = scaled_dot_product_attention(X @ WQ, X @ WK, X @ WV)
    X = layer_norm(X + attn)

    ff = ffn(X, W1, W2)
    X = layer_norm(X + ff)

    return X
