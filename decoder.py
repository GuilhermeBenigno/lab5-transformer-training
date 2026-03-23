import numpy as np
from attention import scaled_dot_product_attention, softmax

D_MODEL = 64
VOCAB_SIZE = 10000


def create_causal_mask(seq_len):
    return np.where(np.tril(np.ones((seq_len, seq_len))), 0.0, -np.inf)


def generate_next_token(current_ids, encoder_output, embedding_table, W_out):
    seq_len = len(current_ids)

    decoder_input = embedding_table[current_ids][np.newaxis, :, :]

    WQ = np.random.randn(D_MODEL, D_MODEL)
    WK = np.random.randn(D_MODEL, D_MODEL)
    WV = np.random.randn(D_MODEL, D_MODEL)

    mask = create_causal_mask(seq_len)

    self_att = scaled_dot_product_attention(
        decoder_input @ WQ,
        decoder_input @ WK,
        decoder_input @ WV,
        mask
    )

    dec_state = self_att + decoder_input

    # Cross Attention
    Q = dec_state @ WQ
    K = encoder_output @ WK
    V = encoder_output @ WV

    cross = scaled_dot_product_attention(Q, K, V)

    last = cross[0, -1, :]

    return softmax(last @ W_out)
