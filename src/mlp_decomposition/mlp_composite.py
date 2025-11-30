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

    def forward(self, model_input: dict[str, torch.Tensor], part_name: str):
        # all coords have the same semantic label
        x = model_input["coords"]
        sdf = self.parts[part_name](x)
        return sdf

    def flatten(self):
        flat_vector = torch.cat([part.flatten() for part in self.parts.values()])
        return flat_vector

    def unflatten(self, flat_vector):
        for part in self.parts.values():
            part.unflatten(flat_vector)


def get_model():
    distribution = {"wing": 0.2, "body": 0.5, "tail": 0.15, "engine": 0.15}
    total_hidden_size = 128

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
            "multires": 10,
            "output_type": "occ",
        }

    model = MLPComposite(registry)

    return model


if __name__ == "__main__":
    model = get_model()
    print(model)
