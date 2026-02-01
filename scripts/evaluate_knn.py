from generate_hybrid_shapes import generate_base_shapes, save_result, load_system

import os
import sys
import argparse
import numpy as np
import torch
import trimesh
import shutil
from tqdm import tqdm

# Add root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "external"))

from src.hd_utils import Config
from external.evaluation_metrics_3d import distChamfer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_gt_dataset(diffuser):
    """Loads all ground truth point clouds."""
    cfg = diffuser.cfg
    dataset_name = cfg.dataset
    pc_dir = os.path.join(cfg.dataset_dir, f"{dataset_name}_{cfg.val.num_points}_pc")
    split_path = os.path.join(cfg.dataset_dir, dataset_name, "train_split.lst")
    
    if not os.path.exists(split_path):
        files = [f.replace('.npy', '') for f in os.listdir(pc_dir) if f.endswith('.npy')]
    else:
        files = np.genfromtxt(split_path, dtype=str)

    print(f"Loading {len(files)} GT point clouds...")
    pcs, names = [], []
    for file_id in tqdm(files):
        path = os.path.join(pc_dir, f"{file_id}.npy")
        if os.path.exists(path):
            data = np.load(path)[:, :3] 
            pcs.append(torch.from_numpy(data).float())
            names.append(file_id)
    return torch.stack(pcs).to(DEVICE), names

def normalize_pc(pc):
    """
    Normalizes point cloud to zero mean and unit variance.
    Matches HyperDiffusion test-time metric calculation.
    """
    # pc shape: (N, 3) or (B, N, 3)
    if pc.ndim == 2:
        shift = pc.mean(dim=0).reshape(1, 3)
        scale = pc.flatten().std().reshape(1, 1)
    else:
        shift = pc.mean(dim=1, keepdim=True)
        scale = pc.flatten(1).std(dim=1, keepdim=True).unsqueeze(-1)
    
    return (pc - shift) / scale

def load_shape_from_file(file_path):
    """Loads a mesh from a specified .ply or .obj file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Specified file not found: {file_path}")
    print(f"Loading external shape from: {file_path}")
    return trimesh.load(file_path, force='mesh')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/diffusion_configs/train_plane_moe.yaml")
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default="knn_results")
    parser.add_argument('--num_gen', type=int, default=5)
    
    parser.add_argument('--mesh_dir', type=str, default="data/baseline/plane", 
                        help="Path to folder containing original .obj files")
    # Currently only works with checkpoint even though loading ply doesnt need it
    parser.add_argument('--load_ply', type=str, default=None, 
                        help="Path to a .ply file to load instead of generating. Overrides num_gen.")
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    diffuser, template_mlp = load_system(args.config, args.ckpt)
    
    gt_tensor, gt_names = load_gt_dataset(diffuser)
    gt_tensor = normalize_pc(gt_tensor)
    
    latents = None
    if args.load_ply:
        print(f"Mode: Loading single file from {args.load_ply}")
        args.num_gen = 1 # Force 1 sample if loading a specific file
    else:
        print(f"Generating {args.num_gen} shapes...")
        latents = generate_base_shapes(diffuser, n_samples=args.num_gen)
    
    # Process Mesh and Sample Points
    gen_pcs = []
    
    print("Processing meshes...")
    for i in range(args.num_gen):
        
        temp_dir = None
        
        if args.load_ply:
            mesh = load_shape_from_file(args.load_ply)
        else:
            latent = latents[i].unsqueeze(0)
            
            temp_dir = os.path.join(args.out_dir, f"temp_{i}")
            os.makedirs(temp_dir, exist_ok=True)
            
            save_prefix = os.path.join(temp_dir, "generated")
            
            save_result(latent, diffuser, save_prefix)
            
            mesh_path = None
            for ext in ['.ply', '.obj']: 
                candidate = save_prefix + ext
                if os.path.exists(candidate):
                    mesh_path = candidate
                    break
            
            if mesh_path is None:
                raise FileNotFoundError(f"No .ply or .obj found at {save_prefix}")
                
            mesh = trimesh.load(mesh_path)
        

        final_query_path = os.path.join(args.out_dir, f"sample_{i}_query.obj")
        mesh.export(final_query_path)
        
        # Sample Points for KNN
        pc_sample = trimesh.sample.sample_surface(mesh, diffuser.cfg.val.num_points)[0]
        pc_sample_tensor = torch.from_numpy(pc_sample).float().to(DEVICE)
        gen_pcs.append(normalize_pc(pc_sample_tensor)) 
        
        # Cleanup temp folder
        if temp_dir is not None and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    gen_pcs = torch.stack(gen_pcs)

    # KNN (Standard Chamfer L2)
    print("Computing Nearest Neighbors...")
    k = 3
    chunk_size = 32
    
    for i in range(args.num_gen):
        query = gen_pcs[i].unsqueeze(0)
        dists_all = []
        
        for j in range(0, len(gt_tensor), chunk_size):
            ref_batch = gt_tensor[j : j + chunk_size]
            d1, d2 = distChamfer(query.expand(ref_batch.shape[0], -1, -1), ref_batch)
            dists_all.append(d1.mean(1) + d2.mean(1))
            
        dists_all = torch.cat(dists_all)
        vals, indices = torch.topk(dists_all, k, largest=False)
        
        sample_out_dir = os.path.join(args.out_dir, f"sample_{i}")
        os.makedirs(sample_out_dir, exist_ok=True)
        
        # Move query into subfolder
        src_query = os.path.join(args.out_dir, f"sample_{i}_query.obj")
        if os.path.exists(src_query):
            shutil.move(src_query, os.path.join(sample_out_dir, "0_query.obj"))
        
        print(f"Sample {i} Neighbors:")
        for rank in range(k):
            idx = indices[rank].item()
            dist = vals[rank].item()
            name = gt_names[idx]
            
            print(f"  Rank {rank+1}: {name} (CD: {dist:.6f})")
            
            original_mesh_path = os.path.join(args.mesh_dir, f"{name}")
            target_path = os.path.join(sample_out_dir, f"{rank+1}_neighbor_{name}_dist_{dist:.4f}.obj")
            
            if os.path.exists(original_mesh_path):
                # Copy the High-Res GT Mesh
                shutil.copy(original_mesh_path, target_path)
            else:
                # Fallback: Save Point Cloud if mesh file is missing
                print(f"    Warning: {original_mesh_path} not found. Saving point cloud.")
                trimesh.points.PointCloud(gt_tensor[idx].cpu().numpy()).export(target_path)

if __name__ == "__main__":
    main()