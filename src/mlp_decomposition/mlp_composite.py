import os
import sys

import torch.nn as nn
import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)
from src.mlp_models import MLP3D


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

    

def get_model(cfg, output_type="occ"):
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

    model = MLPComposite(registry)
    return model


if __name__ == "__main__":
    model = get_model()
    print(model)
