import os
import sys

import torch.nn as nn
import torch
from torch.func import functional_call, stack_module_state, vmap

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)
from src.mlp_models import MLP3D


class MLPMoE(nn.Module):
    def __init__(self, registry):
        super().__init__()
        self.registry = registry
        self.parts = nn.ModuleDict()

        for part_name, part_config in registry.items():
            self.parts[part_name] = MLP3D(**part_config)

        # Cache keys to ensure order consistency between stack and dict reconstruction
        self.part_names = list(registry.keys())

    def forward(self, model_input):
        if isinstance(model_input, dict):
            x_in = model_input["coords"]
            expert_input = model_input
        else:
            x_in = model_input
            expert_input = {"coords": model_input}

        # Manual Stacking (Guarantees Autograd Connection)
        # first expert serves as the architecture template
        first_part_key = self.part_names[0]
        base_model = self.parts[first_part_key]

        # Stack parameters: explicitly connect self.parts[i] to the stacked tensor
        params = {}
        for name, param in base_model.named_parameters():
            # Iterate through all experts and stack this specific parameter
            # This ensures gradients flow from 'params' back to 'self.parts'
            params[name] = torch.stack(
                [self.parts[key].get_parameter(name) for key in self.part_names]
            )

        # Stack buffers: (e.g., if your Embedder has fixed frequency bands)
        buffers = {}
        for name, buffer in base_model.named_buffers():
            buffers[name] = torch.stack(
                [self.parts[key].get_buffer(name) for key in self.part_names]
            )

        # Vectorized Execution
        def compute_expert(p, b, data):
            # functional_call applies the stacked weights 'p' to the 'base_model' architecture
            out = functional_call(base_model, (p, b), args=(data,), kwargs=None)
            return out["model_out"]

        # vmap: (Num_Experts, ...) -> (Num_Experts, Batch, Output)
        expert_results = vmap(compute_expert, in_dims=(0, 0, None))(
            params, buffers, expert_input
        )

        part_outputs = {
            name: expert_results[i] for i, name in enumerate(self.part_names)
        }
        # Max aggregation
        max_occ = expert_results.max(dim=0).values

        return {"model_in": x_in, "model_out": max_occ, "parts": part_outputs}

    def flatten(self):
        # Unchanged
        flat_vector = torch.cat([part.flatten() for part in self.parts.values()])
        return flat_vector

    def unflatten(self, flat_vector):
        # Unchanged
        for part in self.parts.values():
            part.unflatten(flat_vector)

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPMoE_old(nn.Module):
    def __init__(self, registry):
        super().__init__()
        self.parts = nn.ModuleDict()
        self.registry = registry

        for part_name, part_config in registry.items():
            self.parts[part_name] = MLP3D(**part_config)

    def forward(self, model_input):
        # TODO: occ should be label
        # During training, model_input is a dict with "coords" and "occ" keys
        if isinstance(model_input, dict):
            x_in = model_input["coords"]
            expert_input = model_input
        else:
            x_in = model_input
            expert_input = {"coords": model_input}

        # Always run all parts but choose max for inference and give all part outputs for training
        # TODO: For now we loop but stacking the weights for vectorized training would be nice.
        # Problem: This might be inefficient for heterogenous sizes (Block Diagonal?)
        part_outputs = {}
        expert_tensors = []
        for part_name, part_expert in self.parts.items():
            part_output = part_expert(expert_input)["model_out"]
            part_outputs[part_name] = part_output
            expert_tensors.append(part_output)

        # TODO: For optimization you could remove this while training
        # But it could also be used for a global loss
        max_occ = torch.stack(expert_tensors).max(dim=0).values
        return {"model_in": x_in, "model_out": max_occ, "parts": part_outputs}

    def flatten(self):
        flat_vector = torch.cat([part.flatten() for part in self.parts.values()])
        return flat_vector

    def unflatten(self, flat_vector):
        for part in self.parts.values():
            part.unflatten(flat_vector)

    def num_trainable_parameters(self):
        """
        Calculates the total number of trainable parameters
        across all sub-networks in the composite model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPComposite(nn.Module):
    def __init__(self, registry):
        super().__init__()
        self.parts = nn.ModuleDict()
        self.registry = registry

        for part_name, part_config in registry.items():
            self.parts[part_name] = MLP3D(**part_config)

    def forward(self, model_input, part_name=None):
        # During training, model_input is a dict with "coords" and "occ" keys
        if isinstance(model_input, dict):
            x = model_input["coords"]
            semantic_label = model_input.get("semantic_label")
        # During inference, model_input is a tensor
        else:
            x = model_input
            semantic_label = None

        # If part_name is provided (during training), only run that part
        if part_name is not None:
            if part_name in self.parts:
                if isinstance(model_input, dict):
                    return self.parts[part_name](model_input)
                else:
                    return self.parts[part_name]({"coords": model_input})
            else:
                raise ValueError(f"Part {part_name} not found in model.")

        part_sdfs = {}
        min_sdf = None
        for name, network in self.parts.items():
            if isinstance(model_input, dict):
                out = network(model_input)
            else:
                out = network({"coords": model_input})
            part_sdfs[name] = out

            sdf = out["model_out"]
            if min_sdf is None:
                min_sdf = sdf
            else:
                min_sdf = torch.min(min_sdf, sdf)

        return {"model_out": min_sdf, "parts": part_sdfs}

    def flatten(self):
        flat_vector = torch.cat([part.flatten() for part in self.parts.values()])
        return flat_vector

    def unflatten(self, flat_vector):
        for part in self.parts.values():
            part.unflatten(flat_vector)

    def num_trainable_parameters(self):
        """
        Calculates the total number of trainable parameters
        across all sub-networks in the composite model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(cfg, model_type="moe", output_type="occ"):
    distribution = cfg.part_distribution
    part_mlp_config = cfg.part_mlp_config

    num_layers = part_mlp_config["num_layers"]
    total_hidden_size = part_mlp_config["total_hidden_size"]
    input_size = part_mlp_config["input_size"]
    output_size = part_mlp_config["output_size"]
    multires = part_mlp_config["multires"]
    use_leaky_relu = part_mlp_config["use_leaky_relu"]
    use_bias = part_mlp_config["use_bias"]

    registry = {}
    for part_name in distribution.keys():
        hidden_neurons_per_layer = int(total_hidden_size * distribution[part_name])
        # ensure minimum hidden neurons per layer
        hidden_neurons_per_layer = max(16, hidden_neurons_per_layer)

        registry[part_name] = {
            "input_size": input_size,
            "hidden_neurons": [hidden_neurons_per_layer] * num_layers,
            "out_size": output_size,
            "use_leaky_relu": use_leaky_relu,
            "use_bias": use_bias,
            "multires": multires,
            "output_type": output_type,
        }
    if model_type == "moe":
        return MLPMoE(registry)
    elif model_type == "composite":
        return MLPComposite(registry)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_model_no_config(model_type="moe", output_type="occ"):
    """
    Mostly here for backwards compatability. Does not need a config but
    hardcoded values. It is currently used in jupyter notebooks.
    """
    # distribution = {"wing": 0.2, "body": 0.5, "tail": 0.15, "engine": 0.15}
    distribution = {"body": 0.45, "wing": 0.33, "tail": 0.13, "engine": 0.09}
    total_hidden_size = 212

    registry = {}
    for part_name in distribution.keys():
        num_layers = 3
        hidden_neurons_per_layer = int(total_hidden_size * distribution[part_name])
        # ensure that the hidden neurons per layer
        hidden_neurons_per_layer = max(8, hidden_neurons_per_layer)

        registry[part_name] = {
            "input_size": 3,
            "hidden_neurons": [hidden_neurons_per_layer] * num_layers,
            "out_size": 1,
            "use_leaky_relu": False,
            "use_bias": True,
            "multires": 4,
            "output_type": output_type,
        }
    if model_type == "moe":
        return MLPMoE(registry)
    elif model_type == "composite":
        return MLPComposite(registry)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    model = get_model_no_config()
    model_in = torch.zeros(3)
    model_out_train = model.forward(model_in)
    model_out_inf = model.forward({"coords": torch.zeros(3)})
    print(model)
    print(f"model_in: {model_in}")
    print(f"model_out: {model_out_inf}")
