import os
import sys
import logging
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig, open_dict
import hydra
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.dataset import SemanticPointCloud
from src.mlp_decomposition.mlp_composite import get_model
from src.mlp_decomposition.loss import all_part_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_shape_id_from_filename(filename):
    parts = filename.split('_')
    for part in parts:
        if len(part) > 8 and any(char.isdigit() for char in part):
            return part
    return parts[1]

def evaluate_shape(model, shape_id, cfg, labels_map):
    pc_path = os.path.join(cfg.dataset_folder, f"{shape_id}.obj")
    expert_path = os.path.join(cfg.dataset_expert_folder, f"{shape_id}.pts")
    label_path = os.path.join(cfg.label_folder, f"{shape_id}.seg")

    if not os.path.exists(label_path):
        return True, {}

    try:
        # Load dataset exactly as in training
        dataset = SemanticPointCloud(
            on_surface_points=cfg.batch_size, 
            pointcloud_path=pc_path,
            pointcloud_expert_path=expert_path,
            label_path=label_path,
            output_type=cfg.output_type,
            cfg=cfg 
        )
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        try:
            batch = next(iter(loader))
        except StopIteration:
             return True, {}

        model_input, gt = batch
        model_input = {k: v.to(DEVICE) for k, v in model_input.items()}
        gt = {k: v.to(DEVICE) for k, v in gt.items()}

        with torch.no_grad():
            output = model(model_input)
            # This loss function returns the SUM of losses over the batch
            losses = all_part_loss(output, gt, labels_map)
        
        # Return the raw, summed losses for direct comparison
        return False, losses['part_losses']

    except Exception as e:
        logger.error(f"Error evaluating {shape_id}: {e}")
        return True, {}

@hydra.main(version_base=None, config_path="../../configs/overfitting_configs", config_name="overfit_plane_equal")
def main(cfg: DictConfig):
    
    # Based on total loss
    default_threshold = 50.0
    
    threshold = getattr(cfg, 'outlier_threshold', default_threshold) 
    weights_dir = getattr(cfg, 'weights_dir', 'mlp_weights/overfit_plane')
    output_file = getattr(cfg, 'output_file', 'problematic_shapes_filtered.txt')

    logger.info(f"--- Outlier Filtering (Summed Loss) ---")
    logger.info(f"Weights Directory: {weights_dir}")
    logger.info(f"Loss Threshold: {threshold} (Per Part, Sum over {cfg.batch_size} points)")
    
    with open_dict(cfg):
        cfg.mlp_config.output_type = cfg.output_type
        cfg.strategy = "load_pc"
        cfg.augment_on_the_fly = False
        cfg.mesh_jitter = False
        
    model = get_model(cfg).to(DEVICE)
    model.eval()
    
    label_names = cfg.label_names
    labels_map = {name: i + 1 for i, name in enumerate(label_names)}

    if not os.path.exists(weights_dir):
        logger.error(f"Weights directory not found: {weights_dir}")
        return

    weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth') and 'model_final' in f]
    weight_files.sort()
    
    outliers = []
    all_losses = [] # Store losses for histogramm
    
    logger.info(f"Scanning {len(weight_files)} shapes...")
    
    pbar = tqdm(weight_files)
    for fname in pbar:
        shape_id = get_shape_id_from_filename(fname)
        ckpt_path = os.path.join(weights_dir, fname)
        
        try:
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Corrupt checkpoint {fname}: {e}")
            outliers.append(shape_id)
            continue

        is_error, part_losses = evaluate_shape(model, shape_id, cfg, labels_map)
        
        if is_error:
            outliers.append(shape_id)
            continue
            
        is_bad = False
        bad_parts = []
        
        for part, loss_val in part_losses.items():
            loss_val = float(loss_val)
            all_losses.append(loss_val)
            
            if loss_val > threshold:
                is_bad = True
                bad_parts.append(f"{part}:{loss_val:.1f}")
        
        if is_bad:
            outliers.append(shape_id)
            pbar.set_description(f"Outlier: {shape_id} ({bad_parts[0]})")
        else:
            pbar.set_description(f"Processing")

    logger.info(f"Finished. Found {len(outliers)} outliers out of {len(weight_files)}.")
    
    # Save Outlier IDs
    with open(output_file, 'w') as f:
        for oid in outliers:
            f.write(f"{oid}\n")
    logger.info(f"Outlier IDs saved to {output_file}")

    # Histogram plotting
    if all_losses:
        plt.figure(figsize=(10, 6))
        plt.hist(all_losses, bins=50, color='skyblue', edgecolor='black', alpha=0.7, log=False)
        plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Cutoff Threshold: {threshold}')
        
        plt.title(f"Distribution of Reconstruction Errors per Semantic Part (N={len(all_losses)})")
        plt.xlabel("Summed BCE Loss")
        plt.ylabel("Frequency)")
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        
        hist_path = output_file.replace('.txt', '_hist.png')
        if hist_path == output_file: hist_path += ".png" # Fallback if no extension
        
        plt.savefig(hist_path)
        logger.info(f"Error distribution histogram saved to {hist_path}")

if __name__ == "__main__":
    main()