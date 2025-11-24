"""Estimate Transformer Model Parameters and Training Memory Usage"""


def estimate_transformer(n_embd, n_layer, n_mlp_flattened, n_input_tokens=0):
    params_blocks = 12 * n_layer * (n_embd**2)
    params_encdec = 2 * n_mlp_flattened * n_embd
    params_pos = n_input_tokens * n_embd
    params_total = params_blocks + params_encdec + params_pos
    bytes_float32 = params_total * 4  # weights only
    bytes_training_float32 = params_total * 16  # weights + grads + 2 * optimizer slots
    gib_training = bytes_training_float32 / (1024**3)
    gib_weights = bytes_float32 / (1024**3)

    return {
        "params": f"{params_total / 1e6:.2f}M",
        "train_memory_gib_float32": f"{gib_training:.2f}GiB",
        "weights_memory_gib_float32": f"{gib_weights:.2f}GiB",
    }


if __name__ == "__main__":
    # Head dimension time the number of heads equals the embedding dimension
    print(estimate_transformer(n_embd=2880, n_layer=12, n_mlp_flattened=36737))
    print(estimate_transformer(n_embd=1024, n_layer=12, n_mlp_flattened=36737))
