"""
Generates 3D meshes from a trained MLPComposite model checkpoint.

This script produces two types of outputs:
1. A single mesh representing the final, combined shape.
2. A separate mesh for each individual semantic part learned by the sub-networks.
"""

import os
import sys
from pathlib import Path
import torch
import configargparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from external.siren import sdf_meshing, utils
from src.mlp_decomposition.mlp_composite import get_model


class SDFDecoder(torch.nn.Module):
    """
    A wrapper class that makes the MLPComposite model compatible with the sdf_meshing script.
    It can be configured to output either the combined shape or a single part's shape.
    """
    def __init__(self, checkpoint_path, device, cfg, output_type="occ", part_name=None):
        super().__init__()
        self.model = get_model(cfg, output_type=output_type)
        
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        self.model.to(device)
        self.device = device
        self.part_name = part_name

    def forward(self, coords):
        # The meshing script provides a [N, 3] tensor. The model expects [B, N, 3].
        model_in = {"coords": coords.unsqueeze(0)}
        model_out = self.model(model_in, part_name=self.part_name)
        return model_out["model_out"]


def parse_arguments():
    """Parses command-line arguments."""
    p = configargparse.ArgumentParser()
    p.add("-c", "--config_filepath", required=False, is_config_file=True, help="Path to config file.")
    
    p.add_argument("--logging_root", type=str, default="./logs", help="Root directory for logging.")
    p.add_argument("--experiment_name", type=str, required=True, help="Name of subdirectory for saving meshes.")
    p.add_argument("--checkpoint_path", required=True, help="Path to the trained composite model checkpoint.")
    p.add_argument("--resolution", type=int, default=256, help="Resolution for the marching cubes algorithm.")
    p.add_argument("--level", type=float, default=0.0, help="Isosurface level (e.g., 0.0 for occupancy logits).")
    p.add_argument("--output_type", type=str, default="occ", help="Output type of the model (e.g., 'occ' or 'sdf').")
    p.add_argument("--cfg", type=str, default="configs/overfitting_configs/overfit_plane.yaml", help="cfg")

    return p.parse_args()


def generate_combined_mesh(opt, device, output_dir):
    """Generates and saves the mesh for the final, combined shape."""
    print("\n--- Generating mesh for the COMBINED shape ---")
    
    # When part_name is None, the SDFDecoder returns the torch.min of all parts.
    full_decoder = SDFDecoder(opt.checkpoint_path, device, output_type=opt.output_type, part_name=None)
    
    mesh_name = Path(opt.checkpoint_path).stem
    output_path = os.path.join(output_dir, f"{mesh_name}_combined")
    
    sdf_meshing.create_mesh(
        full_decoder,
        output_path,
        N=opt.resolution,
        level=opt.level,
    )
    print(f"Saved combined mesh to '{output_path}.ply'")


def generate_part_meshes(opt, device, output_dir):
    """Generates and saves a separate mesh for each individual part."""
    print("\n--- Generating meshes for INDIVIDUAL parts ---")
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(opt.cfg)

    # Instantiate a temporary model to get the list of part names
    temp_model = get_model(output_type=opt.output_type)
    part_names = list(temp_model.registry.keys())
    mesh_name_stem = Path(opt.checkpoint_path).stem

    for part_name in part_names:
        print(f"Processing part: '{part_name}'...")
        
        # Create a new decoder instance that is locked to a specific part
        part_decoder = SDFDecoder(opt.checkpoint_path, device, cfg, output_type=opt.output_type, part_name=part_name)
        
        output_path = os.path.join(output_dir, f"{mesh_name_stem}_part_{part_name}")
        
        sdf_meshing.create_mesh(
            part_decoder,
            output_path,
            N=opt.resolution,
            level=opt.level,
        )
        print(f"Saved part mesh to '{output_path}.ply'")


def main():
    """Main execution function."""
    opt = parse_arguments()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    output_dir = os.path.join(opt.logging_root, opt.experiment_name)
    utils.cond_mkdir(output_dir)

    # Generate the mesh for the full shape
    # TODO: Fix this as it currently crashes
    #generate_combined_mesh(opt, DEVICE, output_dir)
    
    # Generate a mesh for each part
    generate_part_meshes(opt, DEVICE, output_dir)


if __name__ == "__main__":
    main()