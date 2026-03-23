import numpy as np
from encoder import run_encoder
from decoder import generate_next_token

D_MODEL = 64
VOCAB_SIZE = 10000

TOKEN_START = 0
TOKEN_EOS = 1


def run_transformer():
    embedding_table = np.random.randn(VOCAB_SIZE, D_MODEL)
    W_out = np.random.randn(D_MODEL, VOCAB_SIZE)

    # Entrada simulada
    input_ids = [2, 3, 4]

    # Encoder
    encoder_output = run_encoder(input_ids, embedding_table)
    print("Encoder output shape:", encoder_output.shape)

    # Decoder loop
    current_ids = [TOKEN_START]

    while True:
        probs = generate_next_token(current_ids, encoder_output, embedding_table, W_out)

        next_id = int(np.argmax(probs))
        current_ids.append(next_id)

        print("Token gerado:", next_id)

        if next_id == TOKEN_EOS:
            break

    print("Sequência final:", current_ids)


if __name__ == "__main__":
    run_transformer()
