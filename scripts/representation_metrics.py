import os
import sys
import json
import torch
import numpy as np
import trimesh
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../external")))

from src.hd_utils import generate_mlp_from_weights, calculate_fid_3d
from external.evaluation_metrics_3d import compute_all_metrics
from external.siren import sdf_meshing
from src.mlp_decomposition.test_mlp import SDFDecoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_weight_file(weights_dir, shape_id):
    """Finds the .pth file for a given shape ID in the weights directory."""
    shape_id = shape_id.replace(".obj", "")
    # Pattern 1: Original Hyperdiffusion
    p1 = os.path.join(weights_dir, f"occ_{shape_id}_jitter_0_model_final.pth")
    if os.path.exists(p1): return p1
    
    # Pattern 2: Semantic Hyperdiffusion
    p2 = os.path.join(weights_dir, f"occ_{shape_id}_model_final.pth")
    if os.path.exists(p2): return p2
    
    print(f"Could not find weight id {shape_id}")
    return None

def flatten_weights(state_dict):
    """Flattens weights exactly as done in WeightDataset."""
    weights = []
    for key in state_dict:
        weights.append(state_dict[key].flatten().cpu())
    return torch.hstack(weights)

def generate_mesh_from_weights(weights, mlp_kwargs, res=256):
    """Reconstructs mesh from flattened weights using Marching Cubes."""
    mlp = generate_mlp_from_weights(weights, mlp_kwargs)
    
    sdf_decoder = SDFDecoder(
        checkpoint_path=None, # No checkpoint path, we loaded weights manually
        cfg=mlp_kwargs,
        device=DEVICE
    )
    sdf_decoder.model = mlp.to(DEVICE).eval()
    
    # Marching Cubes
    try:
        verts, faces, _ = sdf_meshing.create_mesh(
            sdf_decoder,
            filename=None,
            N=res,
            level=0.0, 
            device=DEVICE
        )
        return trimesh.Trimesh(vertices=verts, faces=faces)
    except Exception as e:
        print(f"Meshing failed: {e}")
        return None

@hydra.main(version_base=None, config_path="../configs/diffusion_configs", config_name="train_plane_moe")
def main(cfg: DictConfig):
    print(f"--- Evaluating Representation Quality ---")
    print(f"Comparison: Overfitted Weights vs. Ground Truth Point Clouds")
    
    #TODO: Make this same as normal script
    dataset_path = os.path.join(cfg.dataset_dir, cfg.dataset + "_2048_pc") # Using 2048 PC for metrics
    split_file = os.path.join(dataset_path, "test_split.lst")
    
    # Use the training weights folder defined in the diffusion config
    weights_dir = cfg.mlps_folder_train 
    
    if not os.path.exists(weights_dir):
        print(f"Error: Weights directory not found: {weights_dir}")
        return

    all_test_ids = np.genfromtxt(split_file, dtype="str")
    print(f"Total Test Shapes: {len(all_test_ids)}")

    # Only IDs that exist in both GT and Weights folder
    valid_ids = []
    valid_weight_paths = []
    
    print("Filtering shapes...")
    for sid in all_test_ids:
        # Check GT
        gt_path = os.path.join(dataset_path, f"{sid}.npy")
        if not os.path.exists(gt_path): continue
            
        # Check Weights
        w_path = get_weight_file(weights_dir, sid)
        if w_path is None: continue
            
        valid_ids.append(sid)
        valid_weight_paths.append(w_path)
        
    print(f"Valid Shapes (Found in both GT and Weights): {len(valid_ids)}")

    ref_pcs = []   # GT
    sample_pcs = [] # Overfitted Weights
    
    # Dummy logger class for metrics
    class Logger:
        def log(self, *args, **kwargs): pass
        @property
        def experiment(self): return self
    logger = Logger()

    print("Generating Meshes from Weights...")
    for i in tqdm(range(len(valid_ids))):
        sid = valid_ids[i]
        
        pc = np.load(os.path.join(dataset_path, f"{sid}.npy"))[:, :3]
        
        # GT Normalization
        pc = torch.tensor(pc).float()
        shift = pc.mean(dim=0).reshape(1, 3)
        scale = pc.flatten().std().reshape(1, 1)
        pc = (pc - shift) / scale
        ref_pcs.append(pc)

        # Reconstruct
        state_dict = torch.load(valid_weight_paths[i], map_location='cpu')
        flat_weights = flatten_weights(state_dict)
        
        mesh = generate_mesh_from_weights(flat_weights, cfg.mlp_config.params, res=256)
        
        if mesh is None or len(mesh.vertices) == 0:
            print(f"Warning: Empty mesh for {sid}")
            sample_pcs.append(torch.zeros(2048, 3)) # Dummy to keep alignment
        else:
            # Sample 2048 points from mesh
            pts = mesh.sample(2048)
            pts = torch.tensor(pts).float()
            
            # Sample Normalization
            shift = pts.mean(dim=0).reshape(1, 3)
            scale = pts.flatten().std().reshape(1, 1)
            pts = (pts - shift) / scale
            
            sample_pcs.append(pts)

    ref_pcs = torch.stack(ref_pcs).to(DEVICE)
    sample_pcs = torch.stack(sample_pcs).to(DEVICE)

    print("Calculating Metrics (CD, EMD, 1-NNA)...")
    metrics = compute_all_metrics(
        sample_pcs,
        ref_pcs,
        batch_size=16,
        logger=logger
    )

    print("Calculating FPD...")
    fid = calculate_fid_3d(sample_pcs, ref_pcs, logger)
    metrics['FPD'] = fid.item()

    print("\n" + "="*30)
    print("REPRESENTATION QUALITY RESULTS")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")
    print("="*30)

    out_file = "representation_quality_metrics.json"
    with open(out_file, 'w') as f:
        json_metrics = {k: float(v) if torch.is_tensor(v) else v for k,v in metrics.items()}
        json.dump(json_metrics, f, indent=4)
    print(f"Saved metrics to {out_file}")

if __name__ == "__main__":
    main()