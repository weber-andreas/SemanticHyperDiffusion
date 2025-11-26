import os
import sys

import torch.nn as nn
import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)
from src.embedder import Embedder


class MLPBudgetAllocator:
    def __init__(self, total_params, parts_priority, input_dim=3, global_out_dim=1):
        self.total_params = total_params
        self.input_dim = input_dim
        self.global_out_dim = global_out_dim
        self.parts_priority = list(parts_priority.items())

        # Search space
        self.possible_depths = [2, 3, 4, 5, 6]
        self.possible_multires = [0, 2, 4, 6, 8]

    def _get_input_size(self, multires):
        d = self.input_dim
        if multires < 1:
            return d + d
        return d + (multires * 2 * d)

    def calculate_cost(self, width, depth, multires):
        """Calculates exact parameter count for a given architecture."""
        in_dim = self._get_input_size(multires)
        count = (in_dim * width) + width  # Input Layer
        if depth > 1:
            count += (width * width + width) * (depth - 1)  # Hidden Layers
        count += (width * self.global_out_dim) + self.global_out_dim  # Output Layer
        return count

    def find_initial_arch(self, target_budget):
        """Standard binary search to find a 'safe' starting point."""
        best_config = None
        best_count = -1

        for d in self.possible_depths:
            for m in self.possible_multires:
                # Binary search for max even width
                low, high = 2, 1024
                best_w = 0
                while low <= high:
                    mid = (low + high) // 2
                    if mid % 2 != 0:
                        mid -= 1
                    if mid < 2:
                        mid = 2

                    cost = self.calculate_cost(mid, d, m)
                    if cost <= target_budget:
                        best_w = mid
                        low = mid + 2
                    else:
                        high = mid - 2

                if best_w > 0:
                    cost = self.calculate_cost(best_w, d, m)
                    if cost > best_count:
                        best_count = cost
                        best_config = {
                            "hidden_neurons": [best_w] * d,
                            "multires": m,
                            "param_count": cost,
                        }
        return best_config

    def generate_registry(self):
        # 1. INITIAL PASS: Conservative Allocation
        configs = {}
        current_total_used = 0

        for part_name, percentage in self.parts_priority:
            target = int(self.total_params * percentage)
            config = self.find_initial_arch(target)

            # Safety fallback
            if config is None:
                config = {
                    "hidden_neurons": [2],
                    "multires": 0,
                    "param_count": self.calculate_cost(2, 1, 0),
                }

            configs[part_name] = config
            current_total_used += config["param_count"]

        # 2. GREEDY TOP-UP PHASE
        remaining_budget = self.total_params - current_total_used

        iteration = 0
        while remaining_budget > 0 and iteration < 1000:
            best_upgrade = None  # (part_name, new_width, cost_increase)
            min_waste = float("inf")
            found_upgrade = False

            # Check every part to see if we can give it +1 or +2 width
            for part_name in configs.keys():
                cfg = configs[part_name]
                curr_w = cfg["hidden_neurons"][0]
                curr_d = len(cfg["hidden_neurons"])
                curr_m = cfg["multires"]
                curr_cost = cfg["param_count"]

                # Try adding 1 to width (Fine-grained filling)
                next_w = curr_w + 1
                new_cost = self.calculate_cost(next_w, curr_d, curr_m)
                diff = new_cost - curr_cost

                # Does this upgrade fit in the remaining budget?
                if diff <= remaining_budget:
                    # We prefer upgrades that use the MOST of the remaining budget
                    # to converge faster, OR strictly fill tightly.
                    waste = remaining_budget - diff
                    if waste < min_waste:
                        min_waste = waste
                        best_upgrade = (part_name, next_w, diff)
                        found_upgrade = True

            # Apply the best upgrade found in this cycle
            if found_upgrade:
                p_name, w, diff = best_upgrade
                configs[p_name]["hidden_neurons"] = [w] * len(
                    configs[p_name]["hidden_neurons"]
                )
                configs[p_name]["param_count"] += diff
                remaining_budget -= diff
            else:
                # No part can grow anymore without exceeding total. We are stuck.
                # (Usually remaining is extremely small here, like < 5 params)
                break

            iteration += 1

        # 3. FINALIZE REGISTRY
        registry = {"parts": {}, "total_params": self.total_params}  # Use Fixed Total
        current_idx = 0

        # Sort by priority order just to keep things neat
        for part_name, _ in self.parts_priority:
            config = configs[part_name]
            count = config["param_count"]

            padding = 0
            if part_name == self.parts_priority[-1][0]:
                padding = self.total_params - (current_idx + count)

            total_alloc = count + padding

            registry["parts"][part_name] = {
                "config": config,
                "memory_layout": {
                    "functional_params": count,
                    "padding_params": padding,
                    "total_allocated": total_alloc,
                },
                "indices": {"start": current_idx, "end": current_idx + total_alloc},
            }
            current_idx += total_alloc

        return registry


class PartNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_neurons = config["hidden_neurons"]
        multires = config["multires"]

        embed_kwargs = {
            "include_input": True,
            "input_dims": 3,
            "max_freq_log2": multires - 1 if multires > 0 else 0,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }

        self.embedder = Embedder(**embed_kwargs)
        input_dim = self.embedder.out_dim

        layers = []
        if len(hidden_neurons) > 0:
            layers.append(nn.Linear(input_dim, hidden_neurons[0]))
            layers.append(nn.SiLU())  # Changed to SiLU as originally requested
            for i in range(len(hidden_neurons) - 1):
                layers.append(nn.Linear(hidden_neurons[i], hidden_neurons[i + 1]))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_neurons[-1], 1))
        else:
            layers.append(nn.Linear(input_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x_embedded = self.embedder.embed(x)
        return self.net(x_embedded)


class CompositePartNet(nn.Module):
    def __init__(self, registry):
        super().__init__()
        self.parts = nn.ModuleDict()
        self.registry = registry

        for part_name, data in registry["parts"].items():
            self.parts[part_name] = PartNet(data["config"])

    def forward(self, model_input, active_parts=None):
        # Extract coords from input dict if it's a dict, otherwise assume it's a tensor
        if isinstance(model_input, dict):
            x = model_input["coords"]
        else:
            x = model_input

        part_sdfs = []
        for name, network in self.parts.items():
            if active_parts is not None and name not in active_parts:
                continue
            part_sdfs.append(network(x))

        if not part_sdfs:
            # Should not happen if active_parts is None or valid
            return {"model_in": x, "model_out": torch.ones(x.shape[0], 1).to(x.device)}

        all_sdfs = torch.cat(part_sdfs, dim=1)
        global_sdf, _ = torch.min(all_sdfs, dim=1, keepdim=True)

        return {"model_in": x, "model_out": global_sdf}

    def flatten(self):
        # Corrected Key: Matches what generate_registry saves
        total_size = self.registry["total_params"]
        device = next(self.parameters()).device
        flat_vector = torch.zeros(total_size, device=device)

        for name, network in self.parts.items():
            reg_entry = self.registry["parts"][name]
            start_idx = reg_entry["indices"]["start"]

            functional_weights = []
            for param in network.parameters():
                functional_weights.append(param.view(-1))

            part_flat = torch.cat(functional_weights)

            # This handles the placement.
            # If the part has 0 padding, it fits perfectly.
            # If it has padding (last part), the zeros initialized above remain zeros.
            func_len = part_flat.numel()
            flat_vector[start_idx : start_idx + func_len] = part_flat

        return flat_vector


def print_model(model: nn.Module):
    nparameters = sum(p.numel() for p in model.parameters())
    print(model)
    print("Total number of parameters: %d" % nparameters)


def example_mlp_decomposition():
    distribution = {"wing": 0.2, "body": 0.5, "tail": 0.15, "engine": 0.15}
    target_params = 4 + 128 * 3 + 1

    allocator = MLPBudgetAllocator(target_params, distribution)
    result = allocator.generate_registry()

    # --- VALIDATION ---
    print(f"Requested Total: {target_params}")
    print(f"Calculated Total: {result['total_params']}")
    print(f"Match: {target_params == result['total_params']}")
    print("=" * 80)
    print(
        f"{'PART':<10} | {'ALLOCATED':<10} | {'FUNCTIONAL':<10} | {'PADDING':<8} | {'STRUCTURE'}"
    )
    print("-" * 80)

    sorted_parts = sorted(
        result["parts"].items(), key=lambda x: x[1]["indices"]["start"]
    )

    for part, data in sorted_parts:
        allocated = data["memory_layout"]["total_allocated"]
        func = data["memory_layout"]["functional_params"]
        pad = data["memory_layout"]["padding_params"]

        cfg = data["config"]
        struct = f"{cfg['hidden_neurons'][0]}w x {len(cfg['hidden_neurons'])}L (mRes:{cfg['multires']})"

        print(f"{part:<10} | {allocated:<10} | {func:<10} | {pad:<8} | {struct}")


def example_mlp_allocation():
    distribution = {"wing": 0.2, "body": 0.5, "tail": 0.15, "engine": 0.15}
    target_size = 3 * 128

    # 1. Run the Allocator
    allocator = MLPBudgetAllocator(
        total_params=target_size,
        parts_priority=distribution,
    )
    registry = allocator.generate_registry()

    # 2. Instantiate the Decomposed Model
    model = CompositePartNet(registry)
    print(f"\n--- MODEL ARCHITECTURE ---")
    print(model)

    # 3. Flatten the model
    flat_vector = model.flatten()

    print(f"\n--- RESULT ---")
    print(f"Actual Vector Size: {flat_vector.shape[0]}")
    print(f"Is Exact Match?   {flat_vector.shape[0] == target_size}")

    # 4. Inspect the vector content
    print(f"\n--- INSPECTION ---")
    non_zero = torch.count_nonzero(flat_vector)
    padding = flat_vector.shape[0] - non_zero
    print(f"Functional Weights: {non_zero}")
    print(f"Zero Padding:       {padding}")
    print(f"Sparsity:           {padding / flat_vector.shape[0]:.2%}")


if __name__ == "__main__":
    # example_mlp_allocation()
    example_mlp_allocation()
