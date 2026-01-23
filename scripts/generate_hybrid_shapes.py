import argparse
import os
import sys
import random
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.hyperdiffusion import HyperDiffusion
from src.transformer import Transformer
from src.mlp_decomposition.mlp_composite import get_model
from src.hd_utils import Config, generate_mlp_from_weights
from external.siren import sdf_meshing
from src.mlp_decomposition.test_mlp import SDFDecoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_system(config_path, ckpt_path):
    """
    Initializes the HyperDiffusion system. 
    Reconstructs the Transformer architecture based on the MLP topology defined in the config.
    """
    cfg = OmegaConf.load(config_path)
    Config.config = cfg
    
    # Instantiate a dummy MLP for the size of flattened vectors for the Transformer
    template_mlp = get_model(cfg.mlp_config.params, model_type="moe", output_type="occ")
    state_dict = template_mlp.state_dict()
    
    # Transformer expects a list of layer sizes to build its token embeddings
    layers_config = [np.prod(param.shape) for param in state_dict.values()]
    layer_names = list(state_dict.keys())
    total_dim = sum(layers_config)

    transformer = Transformer(
        layers_config, layer_names, **cfg.transformer_config.params
    )

    model = HyperDiffusion.load_from_checkpoint(
        ckpt_path,
        model=transformer,
        train_dt=None, val_dt=None, test_dt=None, # Not needed for inference
        mlp_kwargs=cfg.mlp_config.params,
        image_shape=(1, total_dim),
        method=cfg.method,
        cfg=cfg,
        strict=False
    ).to(DEVICE)
    
    model.eval()
    return model, template_mlp

# SDEEdit Logic
@torch.no_grad()
def generate_base_shapes(diffuser, n_samples=2):
    """Generates random valid shapes from the learned distribution."""
    print(f"Generating {n_samples} parent shapes...")
    samples = diffuser.diff.ddim_sample_loop(
        diffuser.model, 
        (n_samples, *diffuser.image_size[1:])
    )
    return samples

def load_latents_from_file(filepath, normalization_factor):
    """
    Loads MLP weights from disk and flattens them exactly as WeightDataset does.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find weight file: {filepath}")
        
    print(f"Loading weights from: {os.path.basename(filepath)}")
    
    state_dict = torch.load(filepath, map_location=DEVICE)
    
    # Flatten like WeightDataset.get_weights
    weights = []
    for key in state_dict:
        weights.append(state_dict[key].flatten())
    
    flat_weights = torch.hstack(weights)
    
    # Normalize
    latent_vector = flat_weights * normalization_factor
    
    return latent_vector.unsqueeze(0)

def get_part_indices(template_mlp):
    """
    Calculates start/end indices by iterating the state_dict keys.
    This guarantees alignment with the vector produced by load_latents_from_file.
    """
    index_map = {}
    current_idx = 0
    
    state_dict = template_mlp.state_dict()
    
    for key, param in state_dict.items():
        # Extract the part name
        parts_prefix = "parts."
        if key.startswith(parts_prefix):
            # I.e "wing" from "parts.wing.layers.0.weight"
            part_name = key.split(".")[1]
            
            if part_name not in index_map:
                index_map[part_name] = {"start": current_idx, "end": 0}
            
            param_size = param.numel()
            current_idx += param_size
            
            
            index_map[part_name]["end"] = current_idx
            
    return index_map


@torch.no_grad()
def mix_latents(vec_a, vec_b, index_map, parts_from_a):
    """
    Hard stitching of weight vectors using the dynamic index map.
    """
    # Ensure inputs are (1, D)
    if vec_a.dim() == 1: vec_a = vec_a.unsqueeze(0)
    if vec_b.dim() == 1: vec_b = vec_b.unsqueeze(0)

    hybrid = vec_b.clone()
    print(f"Mixing Strategy: Base=B, Overwriting {parts_from_a} from A")

    for part in parts_from_a:
        if part not in index_map:
            raise ValueError(f"Part '{part}' not found in the model architecture.")
        
        start = index_map[part]["start"]
        end = index_map[part]["end"]
        
        print(f"  - Swapping '{part}': indices {start} to {end}")
        
        # tensor[batch_indices, feature_indices]
        hybrid[:, start:end] = vec_a[:, start:end]
        
    return hybrid

@torch.no_grad()
def harmonize(diffuser, latent_vector, strength):
    """
    Applies SDEEdit (Stochastic Differential Editing).
    1. Forward diffuse (perturb) the stitched vector to time t_start.
    2. Reverse diffuse (denoise) back to t=0 to heal discontinuities.
    """
    if strength <= 0.0: return latent_vector
    
    diff = diffuser.diff
    model = diffuser.model
    
    # Calculate start step for edit
    t_start = int(diff.num_timesteps * strength)
    t_start = min(t_start, diff.num_timesteps - 1)
    
    print(f"Harmonizing (Healing) with strength {strength} (t={t_start})...")
    
    # Perturb
    # Create batch of timesteps (size 1)
    t_batch = torch.full((1,), t_start, device=DEVICE, dtype=torch.long)
    
    x_start = latent_vector
    noise = torch.randn_like(x_start)
    
    # q_sample expects x_start and noise to be (B, ...). Here (1, D).
    x_noisy, _ = diff.q_sample(x_start, t_batch, noise=noise)
    
    # Denoise
    img = x_noisy
    iterator = tqdm(reversed(range(0, t_start)), desc="Denoising", total=t_start)
    
    for i in iterator:
        # Create timestep tensor for current step
        t = torch.full((1,), i, device=DEVICE, dtype=torch.long)
        
        out = diff.p_sample(model, img, t, clip_denoised=False)
        img = out["sample"]
        
    return img

def save_result(weights, diffuser, filename):
    """Denormalizes weights, saves .pth, and generates .ply mesh."""
    # Denormalize
    final_weights = weights / diffuser.cfg.normalization_factor
    
    # Remove batch dimension [1, Total_Dim] -> [Total_Dim]
    if final_weights.dim() > 1:
        final_weights = final_weights.squeeze()

    model_config = diffuser.mlp_kwargs
    mlp = get_model(model_config, model_type="moe", output_type="occ")
    
    # Manually load weights into the Composite model
    state_dict = mlp.state_dict()
    current_idx = 0
    
    for key in state_dict:
        param = state_dict[key]
        flat_size = param.numel()
        
        chunk = final_weights[current_idx : current_idx + flat_size]
        
        state_dict[key] = chunk.view(param.shape)
        current_idx += flat_size
        
    mlp.load_state_dict(state_dict)

    torch.save(mlp.state_dict(), f"{filename}.pth")
    
    # Generate Mesh
    sdf_decoder = SDFDecoder(
        None, cfg=diffuser.mlp_kwargs, device=DEVICE
    )
    sdf_decoder.model = mlp.to(DEVICE)
    
    sdf_meshing.create_mesh(
        sdf_decoder, filename, N=256, level=0.0, device=DEVICE
    )
    print(f"Saved: {filename}.ply")

def main():
    parser = argparse.ArgumentParser(description="Generate Hybrid Shapes via Semantic SDEEdit")
    parser.add_argument('--config', type=str, default="configs/diffusion_configs/train_plane_moe.yaml")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to diffusion .ckpt")
    parser.add_argument('--out_dir', type=str, default="hybrid_results")
    parser.add_argument('--strength', type=float, default=0.2, help="Harmonization strength (0.0 - 1.0)")
    parser.add_argument('--mix_count', type=int, default=2, help="Number of parts to mix from A")
    parser.add_argument('--parts', type=str, nargs='+', default=None, 
                        help="Specific part names or indices to swap. If not provided, random selection is used.")
    parser.add_argument('--shape_A', type=str, default=None, help="Path to .pth weights for Parent A (Source)")
    parser.add_argument('--shape_B', type=str, default=None, help="Path to .pth weights for Parent B (Base)")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    diffuser, template_mlp = load_system(args.config, args.ckpt)
    
    part_index_map = get_part_indices(template_mlp)
    all_parts = list(part_index_map.keys())
    print(f"Computed Vector Layout: {all_parts}")

    norm_factor = diffuser.cfg.normalization_factor
    # Handle Parent A
    if args.shape_A:
        parent_A = load_latents_from_file(args.shape_A, norm_factor)
    else:
        print("Generating Parent A via Diffusion...")
        parent_A = generate_base_shapes(diffuser, n_samples=1)[0].unsqueeze(0)

    if args.shape_B:
        parent_B = load_latents_from_file(args.shape_B, norm_factor)
    else:
        print("Generating Parent B via Diffusion...")
        parent_B = generate_base_shapes(diffuser, n_samples=1)[0].unsqueeze(0)
    
    save_result(parent_A, diffuser, os.path.join(args.out_dir, "parent_A"))
    save_result(parent_B, diffuser, os.path.join(args.out_dir, "parent_B"))

    # Select Random Parts if given
    if args.parts:
        parts_to_swap = []
        for p in args.parts:
            # Check if index or name
            if p.isdigit():
                idx = int(p)
                if 0 <= idx < len(all_parts):
                    parts_to_swap.append(all_parts[idx])
                else:
                    print(f"Warning: Index {idx} out of range. Skipping.")
            elif p in all_parts:
                parts_to_swap.append(p)
            else:
                print(f"Warning: Part name '{p}' not found in layout. Skipping.")
        
        if not parts_to_swap:
            print("No valid parts specified. Falling back to random selection.")
            parts_to_swap = random.sample(all_parts, k=min(args.mix_count, len(all_parts)))
    else:
        # Default: Random Selection
        parts_to_swap = random.sample(all_parts, k=min(args.mix_count, len(all_parts)))

    print(f"Parts being swapped: {parts_to_swap}")
    
    # Naive Stitching
    naive_latent = mix_latents(parent_A, parent_B, part_index_map, parts_to_swap)
    save_result(naive_latent, diffuser, os.path.join(args.out_dir, "hybrid_naive"))

    # SDFEdit version
    healed_latent = harmonize(diffuser, naive_latent, strength=args.strength)
    save_result(healed_latent, diffuser, os.path.join(args.out_dir, "hybrid_healed"))

if __name__ == "__main__":
    main()