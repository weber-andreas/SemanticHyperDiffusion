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


def estimate_flattened_mlp_params(input_size, hidden_sizes, output_size):
    """Estimate the total number of parameters (weights + biases) for a fully connected MLP."""
    sizes = [input_size] + hidden_sizes + [output_size]
    total_params = 0
    for i in range(len(sizes) - 1):
        weights = sizes[i] * sizes[i + 1]
        biases = sizes[i + 1]
        total_params += weights + biases
    return total_params


if __name__ == "__main__":
    n_mlp_flat = estimate_flattened_mlp_params(
        input_size=3, hidden_sizes=[128, 128, 128], output_size=1
    )
    print(f"Flattened MLP params: {n_mlp_flat}")
    n_mlp_flat_moe = 4 * estimate_flattened_mlp_params(
        input_size=3, hidden_sizes=[60, 60, 60], output_size=1
    )
    n_mlp_flat_moe_2 = 4 * estimate_flattened_mlp_params(
        input_size=3, hidden_sizes=[64, 64, 64], output_size=1
    )
    print(f"Flattened MoE MLP params: {n_mlp_flat_moe}")
    print(f"Flattened MoE MLP params v2: {n_mlp_flat_moe_2}")
    # print(estimate_transformer(n_embd=2880, n_layer=12, n_mlp_flattened=n_mlp_flat))

    print(estimate_transformer(n_embd=768, n_layer=12, n_mlp_flattened=n_mlp_flat))
    print(estimate_transformer(n_embd=768, n_layer=12, n_mlp_flattened=n_mlp_flat_moe))
