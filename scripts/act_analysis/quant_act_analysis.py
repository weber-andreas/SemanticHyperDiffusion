import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Needed for Path
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.mlp_models import MLP3D
from src.mlp_decomposition.mlp_composite import get_model as get_composite_model

# Global State & Hook
activations_capture = {}
synset_offset_2_category = {
    "02691156": "Airplane", "02773838": "Bag", "02954340": "Cap", "02958343": "Car", "03001627": "Chair",
    "03261776": "Earphone", "03467517": "Guitar", "03624134": "Knife", "03636649": "Lamp", "03642806": "Laptop",
    "03790512": "Motorbike", "03797390": "Mug", "03948459": "Pistol", "04099429": "Rocket", "04225987": "Skateboard",
    "04379243": "Table"
}

def get_activation(name: str):
    """Factory for PyTorch forward hooks to capture activations."""
    def hook(model, input, output):
        activations_capture[name] = output.detach()
    return hook


def load_activations_single_mlp(mlp_path: str, points_tensor: torch.Tensor, mlp_params: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """Loads a single MLP3D, registers hooks, and captures activations."""
    global activations_capture
    activations_capture.clear()
    model = MLP3D(**mlp_params).to(device)
    model.load_state_dict(torch.load(mlp_path, map_location=device))
    model.eval()
    for i, layer in enumerate(model.layers[:-1]):
        if isinstance(layer, torch.nn.Linear):
            layer.register_forward_hook(get_activation(f'hidden_layer_{i}'))
    with torch.no_grad():
        model({'coords': points_tensor[None, ...]})
    return {name: tensor.squeeze(0).cpu() for name, tensor in activations_capture.items()}

def load_activations_composite(mlp_path: str, points_tensor: torch.Tensor, device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
    """Loads an MLPComposite, registers hooks, and captures activations for each part."""
    global activations_capture
    model = get_composite_model(output_type="occ").to(device)
    model.load_state_dict(torch.load(mlp_path, map_location=device))
    model.eval()
    all_parts_activations = {}
    for part_name, part_model in model.parts.items():
        activations_capture.clear()
        for i, layer in enumerate(part_model.layers[:-1]):
            if isinstance(layer, torch.nn.Linear):
                layer.register_forward_hook(get_activation(f'hidden_layer_{i}'))
        with torch.no_grad():
            model({'coords': points_tensor[None, ...]}, part_name=part_name)
        all_parts_activations[part_name] = {name: tensor.squeeze(0).cpu() for name, tensor in activations_capture.items()}
    return all_parts_activations


#TODO: Maybe do this as an import
def align_and_get_labels(coords_from_activations, label_coords_path, label_seg_path, alignment_threshold, **kwargs) -> Optional[np.ndarray]:
    """Aligns labeled point cloud and remaps labels via nearest neighbors."""
    coords_from_labels_raw = np.loadtxt(label_coords_path)
    original_part_labels = np.loadtxt(label_seg_path, dtype=int)
    def normalize_to_half_box(points):
        mean = np.mean(points, axis=0, keepdims=True)
        points_centered = points - mean
        v_max, v_min = np.amax(points_centered), np.amin(points_centered)
        scale_factor = 0.5 * 0.95 / (max(abs(v_min), abs(v_max)))
        return points_centered * scale_factor
    target_coords_normalized = normalize_to_half_box(coords_from_activations)
    label_coords_normalized = normalize_to_half_box(coords_from_labels_raw)
    rotation = Rotation.from_euler('y', 90, degrees=True)
    aligned_label_coords = rotation.apply(label_coords_normalized)
    nn_search = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(aligned_label_coords)
    distances, indices = nn_search.kneighbors(target_coords_normalized)
    if np.max(distances) > alignment_threshold:
        print(f"  [WARNING] High alignment error ({np.max(distances):.4f}). Skipping analysis.")
        return None
    return original_part_labels[indices.squeeze()]

def calculate_metrics_for_group(activations, point_indices, all_indices) -> Dict:
    """Calculates quantitative metrics, including both Global and Top-5 stats."""
    if len(point_indices) < 2: return {}

    group_pre_acts = activations[point_indices]
    group_post_acts = F.relu(group_pre_acts)

    # Global metrics across all neurons for this group of points
    global_mean_act = group_post_acts.mean().item()
    global_variance = torch.var(group_post_acts, dim=0).mean().item()
    # Sparsity: Percentage of non-zero activations for this group
    sparsity = (group_post_acts > 0).float().mean().item() * 100

    
    # Top-k metrics focused on the most active neurons
    avg_post_acts_per_neuron = torch.mean(group_post_acts, dim=0)
    topk_k = min(5, group_pre_acts.shape[1])
    if topk_k == 0: return {}
    top_indices = torch.topk(avg_post_acts_per_neuron, k=topk_k).indices

    avg_top5_mean = avg_post_acts_per_neuron[top_indices].mean().item()
    top5_pre_acts = group_pre_acts[:, top_indices]
    avg_top5_var = torch.var(top5_pre_acts, dim=0).mean().item()
    
    # Specialization
    outside_indices = np.setdiff1d(all_indices, point_indices)
    specialized_count = 0
    if len(outside_indices) > 0:
        mean_on_rest_post = torch.mean(F.relu(activations[outside_indices]), dim=0)
        is_specialized = avg_post_acts_per_neuron > (2 * mean_on_rest_post)
        specialized_count = torch.sum(is_specialized).item()
        
    return {
        "Sparsity (%)": sparsity, 
        "Avg Top-5 Mean Act": avg_top5_mean,
        "Avg Top-5 Variance": avg_top5_var,
        "Global Mean Act": global_mean_act,
        "Global Variance": global_variance,
        "Specialized Neurons": specialized_count,
        "Top-k Neuron IDs": top_indices.numpy()
    }

def calculate_jaccard_disentanglement(groups: Dict) -> float:
    """Calculates average pairwise Jaccard similarity between groups."""
    if len(groups) < 2: return np.nan
    all_jaccards = []
    group_ids = list(groups.keys())
    for i in range(len(group_ids)):
        for j in range(i + 1, len(group_ids)):
            top_i = set(groups[group_ids[i]]["Top-k Neuron IDs"])
            top_j = set(groups[group_ids[j]]["Top-k Neuron IDs"])
            intersection = len(top_i.intersection(top_j))
            union = len(top_i.union(top_j))
            all_jaccards.append(intersection / union if union > 0 else 0)

    return np.mean(all_jaccards) if all_jaccards else 0.0


def analyze_shape(shape_id, model_type, mlp_params, paths, device, part_label_to_name, alignment_threshold) -> List[Dict]:
    """Main analysis pipeline for a single shape, dispatched by model_type."""
    pc_npy_path = paths['pc_dir'] / f"{shape_id}.obj.npy"
    # TODO: Same jitter issue as above
    mlp_path = paths['mlp_dir'] / f"occ_{shape_id}_model_final.pth"
    pc_pts_path = paths['part_anno_points'] / f"{shape_id}.pts"
    seg_path = paths['part_anno_labels'] / f"{shape_id}.seg"
    
    activations_coords = np.load(pc_npy_path)[:, :3]
    points_tensor = torch.from_numpy(activations_coords).float().to(device)
    part_labels = align_and_get_labels(activations_coords, pc_pts_path, seg_path, alignment_threshold)
    if part_labels is None: return []

    all_results = []
    
    grouping_methods = {
        "Semantic Parts": (part_labels, part_label_to_name),
        "Whole Figure": (np.zeros(len(activations_coords), dtype=int), {0: "Whole Figure"})
    }

    if model_type == 'composite':
        all_parts_activations = load_activations_composite(mlp_path, points_tensor, device)
        part_names = sorted(all_parts_activations.keys())
        layer_names = list(next(iter(all_parts_activations.values())).keys())
        
        activations_dict = {}
        for layer_name in layer_names:
            total_neurons = sum(all_parts_activations[p][layer_name].shape[1] for p in part_names)
            virtual_activations = torch.zeros(len(points_tensor), total_neurons)
            
            offset = 0
            for part_name in part_names:
                part_id = [k for k, v in part_label_to_name.items() if v.lower() == part_name.lower()][0]
                point_indices = np.where(part_labels == part_id)[0]
                
                activations = all_parts_activations[part_name][layer_name]
                num_neurons_part = activations.shape[1]
                
                if len(point_indices) > 0:
                    virtual_activations[point_indices, offset : offset + num_neurons_part] = activations[point_indices]
                
                offset += num_neurons_part
            activations_dict[layer_name] = virtual_activations
    
    elif model_type == 'single':
        activations_dict = load_activations_single_mlp(mlp_path, points_tensor, mlp_params, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # ANALYSIS LOGIC FOR BOTH MODEL TYPES
    for layer_name, layer_activations in sorted(activations_dict.items()):
        all_layer_metrics = defaultdict(dict)
        for group_method, (labels, name_map) in grouping_methods.items():
            for group_id in np.unique(labels):
                point_indices = np.where(labels == group_id)[0]
                metrics = calculate_metrics_for_group(layer_activations, point_indices, np.arange(len(activations_coords)))
                if metrics: all_layer_metrics[group_method][group_id] = metrics
        
        if "Semantic Parts" in all_layer_metrics:
            jaccard_index = calculate_jaccard_disentanglement(all_layer_metrics["Semantic Parts"])
            for group_metrics in all_layer_metrics["Semantic Parts"].values():
                group_metrics['Disentanglement (Avg Jaccard)'] = jaccard_index
        
        if "Whole Figure" in all_layer_metrics:
            jaccard_whole = 0.0 if model_type == 'composite' else (1.0 if "Semantic Parts" in all_layer_metrics else np.nan)
            all_layer_metrics["Whole Figure"][0]['Disentanglement (Avg Jaccard)'] = jaccard_whole
        
        for group_method, groups in all_layer_metrics.items():
            for group_id, metrics in groups.items():
                res = {"Shape ID": shape_id, "Layer": layer_name, "Grouping": group_method, "Group Name": grouping_methods[group_method][1].get(group_id, f"ID {group_id}")}
                res.update(metrics)
                if "Top-k Neuron IDs" in res: del res["Top-k Neuron IDs"]
                all_results.append(res)
                
    return all_results

def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config_path)
    mlp_params = cfg.get('mlp_config')
    if mlp_params: mlp_params['output_type'] = cfg.output_type

    category_name = synset_offset_2_category[args.category_id]
    if category_name == 'Airplane': part_label_to_name = {1: "Body", 2: "Wing", 3: "Tail", 4: "Engine"}
    else: print(f"Warning: Part labels for '{category_name}' not hardcoded."); part_label_to_name = {i: f"Part {i}" for i in range(1, 50)}
    paths = {'pc_dir': Path(args.dataset_dir), 'mlp_dir': Path(args.mlp_weights_dir), 'part_anno_points': Path(args.part_anno_dir) / args.category_id / 'points', 'part_anno_labels': Path(args.part_anno_dir) / args.category_id / 'expert_verified' / 'points_label'}
    
    all_mlp_files = sorted(paths['mlp_dir'].glob('*.pth'))
    shape_ids_to_process = []
    print("Screening dataset for complete file sets...")
    for mlp_file in tqdm(all_mlp_files, desc="Screening files"):
        try: shape_id = mlp_file.name.split('_')[1]
        except IndexError: continue
        #TODO: Make jitter_0 an optional addition for backwards compatability
        # Currently you have to add jitter_0 manually for old mlp weights
        if all(p.exists() for p in [paths['pc_dir'] / f"{shape_id}.obj.npy", paths['mlp_dir'] / f"occ_{shape_id}_model_final.pth", paths['part_anno_points'] / f"{shape_id}.pts", paths['part_anno_labels'] / f"{shape_id}.seg"]):
            shape_ids_to_process.append(shape_id)
            
    if not shape_ids_to_process: print("No complete sets of files were found. Exiting."); return
        
    all_shape_results = []
    print(f"\nFound {len(shape_ids_to_process)} complete shapes. Starting analysis...")
    for shape_id in tqdm(shape_ids_to_process, desc="Analyzing Shapes"):
        results = analyze_shape(shape_id, args.model_type, mlp_params, paths, device, part_label_to_name, args.alignment_threshold)
        all_shape_results.extend(results)

    if not all_shape_results: print("No results were generated. Exiting."); return

    df = pd.DataFrame(all_shape_results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nAnalysis complete. Full results saved to '{args.output_csv}'")
    
    float_formatter = "{:.4f}".format
    
    summary_df_main = df[df['Grouping'] == 'Whole Figure'].copy()
    if not summary_df_main.empty:
        summary_main = summary_df_main.groupby(['Layer']).mean(numeric_only=True)
        print("\n--- Summary of Averages (Overall Model Performance) ---")
        print(summary_main.to_string(float_format=float_formatter))
    
    summary_df_detail = df[df['Grouping'] == 'Semantic Parts'].copy()
    if not summary_df_detail.empty:
        summary_detail = summary_df_detail.groupby(['Group Name', 'Layer']).mean(numeric_only=True)
        print("\n--- Summary of Averages (Per-Part Details) ---")
        print(summary_detail.to_string(float_format=float_formatter))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run quantitative activation analysis over a dataset.")
    parser.add_argument('--model_type', type=str, required=True, choices=['single', 'composite'], help="Type of model to analyze: 'single' (MLP3D) or 'composite'.")
    parser.add_argument('--config_path', type=str, default='configs/overfitting_configs/overfit_plane.yaml', help='Path to the model config file.')
    parser.add_argument('--dataset_dir', type=str, default='./data/baseline/02691156_2048_pc', help='Directory of pre-sampled point clouds (.npy files).')
    parser.add_argument('--category_id', type=str, default='02691156', help='ShapeNet category ID.')
    parser.add_argument('--mlp_weights_dir', type=str, default='./mlp_weights/212_statistic', help='Directory with MLP .pth checkpoints.')
    parser.add_argument('--part_anno_dir', type=str, default='./data/shapenetpart/PartAnnotation', help='Root directory for PartAnnotation data.')
    parser.add_argument('--output_csv', type=str, default='quantitative_analysis_results.csv', help='Path to save the final CSV results.')
    parser.add_argument('--alignment_threshold', type=float, default=0.1, help='Threshold for alignment error during label matching.')

    args = parser.parse_args()
    main(args)