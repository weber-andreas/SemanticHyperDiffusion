import argparse
import os
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.mlp_models import MLP3D
from src.mlp_decomposition.mlp_composite import get_model as get_composite_model

# This global dictionary stores the activations captured by the hooks
activations_capture = {}

def get_activation(name):
    """Hook function to capture the output of a layer during the forward pass."""
    def hook(model, input, output):
        activations_capture[name] = output.detach()
    return hook

def load_and_prepare_points(args, device):
    """Loads and prepares the point cloud data based on configuration."""
    pc_path = os.path.join(args.pc_dir, f"{args.figure_name}.obj.npy")
    if not os.path.exists(pc_path):
        raise FileNotFoundError(f"Point cloud file not found at: {pc_path}")
        
    point_cloud_data = np.load(pc_path)
    
    if args.pc_format == 'normal':
        points = point_cloud_data[:, :3]
        print(f"Loaded {len(points)} surface points (normal format).")
    elif args.pc_format == 'occupancy':
        occupied_mask = point_cloud_data[:, 3] > 0.5
        points = point_cloud_data[occupied_mask][:, :3]
        print(f"Found {len(points)} occupied points out of {len(point_cloud_data)} total points (occupancy format).")
    else:
        raise ValueError(f"Unknown point cloud format: {args.pc_format}.")
        
    if len(points) == 0:
        print("Warning: No points to process.")
        return None

    return torch.from_numpy(points).float().to(device)

def process_single_mlp(args, device):
    """Processes a single MLP3D model."""
    global activations_capture
    
    #TODO: Make this work like quant act analysis
    mlp_path = os.path.join(args.mlp_dir, f"occ_{args.figure_name}_jitter_0_model_final.pth")
    cfg = OmegaConf.load(args.config_path)
    mlp_params = cfg.get('mlp_config')
    with open_dict(mlp_params): mlp_params['output_type'] = cfg.output_type
    
    model = MLP3D(**mlp_params)
    state_dict = torch.load(mlp_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print("Single MLP3D model loaded successfully.")

    for i, layer in enumerate(model.layers[:-1]):
        if isinstance(layer, torch.nn.Linear):
            layer.register_forward_hook(get_activation(f'hidden_layer_{i}'))
            
    points_tensor = load_and_prepare_points(args, device)
    if points_tensor is None: return

    with torch.no_grad():
        _ = model({'coords': points_tensor[None, ...]})
        
    return {name: tensor.squeeze(0).cpu() for name, tensor in activations_capture.items()}

def process_composite_mlp(args, device):
    """Processes an MLPComposite model, capturing activations for each part."""
    global activations_capture
    
    mlp_path = os.path.join(args.mlp_dir, f"occ_{args.figure_name}_model_final.pth")
    model = get_composite_model(output_type="occ")
    state_dict = torch.load(mlp_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print("MLPComposite model loaded successfully.")

    points_tensor = load_and_prepare_points(args, device)
    if points_tensor is None: return

    all_parts_activations = {}
    for part_name, part_model in model.parts.items():
        activations_capture.clear()
        for i, layer in enumerate(part_model.layers[:-1]):
            if isinstance(layer, torch.nn.Linear):
                layer.register_forward_hook(get_activation(f'hidden_layer_{i}'))

        with torch.no_grad():
            _ = model({'coords': points_tensor[None, ...]}, part_name=part_name)

        all_parts_activations[part_name] = {name: tensor.squeeze(0).cpu() for name, tensor in activations_capture.items()}
    
    return all_parts_activations

def main(args):
    """Main function to dispatch based on model type."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"\n--- Analyzing Figure: {args.figure_name} ---")
    
    final_activations = None
    if args.model_type == 'single':
        final_activations = process_single_mlp(args, device)
    elif args.model_type == 'composite':
        final_activations = process_composite_mlp(args, device)
    else:
        raise ValueError(f"Invalid model_type specified: {args.model_type}")

    if final_activations is None:
        print("Activation capture failed. Exiting.")
        return

    # Save the Results
    output_dir = 'activations'
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.join(output_dir, f'{args.figure_name}_{args.model_type}.pt')
    torch.save(final_activations, output_filename)
    print(f"\nSaved the captured activations to '{output_filename}' for further analysis.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze and save neuron activations of an overfitted MLP.")
    
    parser.add_argument('--model_type', type=str, required=True, choices=['single', 'composite'],
                        help="Type of model to analyze.")
    
    parser.add_argument('--figure_name', type=str, 
                        default='e8409b544c626028a9b2becd26dc2fc1',
                        help='The common name/ID of the figure to analyze.')
                        
    parser.add_argument('--mlp_dir', type=str, 
                        default='logs/test_experiment',
                        help='Directory containing the .pth files of the overfitted MLPs.')

    parser.add_argument('--pc_dir', type=str, 
                        default='data/baseline_200000_pc',
                        help='Directory containing the corresponding .npy point cloud files.')

    parser.add_argument('--config_path', type=str,
                        default='configs/overfitting_configs/overfit_plane.yaml',
                        help='Path to the yaml config file that defines the single MLP architecture.')

    parser.add_argument('--pc_format', type=str,
                        default='occupancy',
                        choices=['normal', 'occupancy'],
                        help="Format of the point cloud file: 'occupancy' for [x,y,z,occ] or 'normal' for older formats.")

    args = parser.parse_args()
    main(args)