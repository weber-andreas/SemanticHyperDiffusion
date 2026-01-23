"""
Render 3D meshes from MLP checkpoints with configurable parameters.

This script:
1. Loads an MLP checkpoint and its associated config
2. Reconstructs a mesh using the neural implicit function (SDF/occupancy)
3. Renders the mesh to an image using the render_meshes.py utilities
4. Optionally creates a grid of multiple rendered meshes
"""

import os
import sys
import glob
from pathlib import Path
import torch
import configargparse
import trimesh
from omegaconf import OmegaConf

# Add the parent directory to path to import from src
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "external/"))
sys.path.append(os.path.join(ROOT_DIR, "external/siren/"))


from external.siren import sdf_meshing
from src.mlp_decomposition.test_mlp import SDFDecoder
from scripts.render_meshes import render_mesh_with_ground, create_grid_image

os.environ["PYOPENGL_PLATFORM"] = "egl"
DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def mesh_from_checkpoint(
    checkpoint_path,
    config_path,
    output_dir,
    resolution=256,
    output_type="occ",
    name=None,
):
    """Generate a mesh from an MLP checkpoint using marching cubes."""
    # Load config
    cfg = load_config(config_path)
    print(f"Loaded config from {config_path}")
    # print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate mesh filename
    if name is None:
        checkpoint_stem = Path(checkpoint_path).stem
        name = f"{checkpoint_stem}_mesh"
    mesh_path = os.path.join(output_dir, "ply_files", name)

    # Create decoder and generate mesh
    sdf_decoder = SDFDecoder(
        checkpoint_path,
        device=DEVICE,
        cfg=cfg,
        output_type=output_type,
    )

    print(f"Generating mesh at resolution {resolution}...")
    vertices, faces, sdf = sdf_meshing.create_mesh(
        sdf_decoder,
        mesh_path,
        N=resolution,
        level=0,
        device=DEVICE,
    )

    print(f"Mesh saved to {mesh_path}.ply")
    return f"{mesh_path}.ply"


def render_checkpoint_mesh(
    checkpoint_path,
    config_path,
    output_dir,
    resolution=256,
    output_type="occ",
    name=None,
):
    """
    Generate a mesh from an MLP checkpoint and render it to an image."""
    # Generate mesh
    mesh_path = mesh_from_checkpoint(
        checkpoint_path,
        config_path,
        output_dir,
        resolution=resolution,
        output_type=output_type,
        name=name,
    )

    # Load and render mesh
    print(f"Rendering mesh...")
    mesh = trimesh.load(mesh_path)
    rendered_img = render_mesh_with_ground(mesh, skip_cleanup=False)

    # Save rendered image
    if name is None:
        checkpoint_stem = Path(checkpoint_path).stem
        name = f"{checkpoint_stem}_render"
    img_path = os.path.join(output_dir, f"{name}.png")
    rendered_img.save(img_path)

    print(f"Rendered image saved to {img_path}")
    return img_path


def batch_render_checkpoints(
    checkpoint_dir,
    config_path,
    output_dir,
    resolution=256,
    output_type="occ",
    pattern="*.pth",
    max_checkpoints=None,
    create_grid=False,
    grid_rows=3,
    grid_cols=4,
    grid_pad=5,
    grid_row_gap=-10,
    grid_col_gap=4,
    skip_files=[],
):
    """Render multiple MLP checkpoints in a batch."""

    os.makedirs(output_dir, exist_ok=True)

    # Try to find checkpoints with the specified pattern first
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))

    # If no files found with the specified pattern, try common patterns
    if checkpoint_files:
        checkpoint_files.sort()
    else:
        common_patterns = ["*.pth", "*.pt", "*.ckpt"]
        for p in common_patterns:
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, p))
            if checkpoint_files:
                print(f"Found checkpoints with pattern '{p}'")
                break

    checkpoint_files.sort()
    # Filter out skip files
    filtered_checkpoints = []
    for ckpt in checkpoint_files:
        if Path(ckpt).stem in skip_files:
            print(f"Skipping checkpoint: {Path(ckpt).name}")
            continue
        filtered_checkpoints.append(ckpt)
    checkpoint_files = filtered_checkpoints

    # Limit checkpoints if max_checkpoints is specified
    if max_checkpoints is not None:
        checkpoint_files = checkpoint_files[:max_checkpoints]

    print(f"Found {len(checkpoint_files)} checkpoint(s)")

    rendered_images = []
    for i, ckpt_path in enumerate(checkpoint_files):
        print(f"\n[{i+1}/{len(checkpoint_files)}] Processing {Path(ckpt_path).name}...")
        try:
            # Keep original filename, just append _render
            checkpoint_stem = Path(ckpt_path).stem
            img_path = render_checkpoint_mesh(
                ckpt_path,
                config_path,
                output_dir,
                resolution=resolution,
                output_type=output_type,
                name=f"{checkpoint_stem}_render",
            )
            rendered_images.append(img_path)
        except Exception as e:
            print(f"Error processing {ckpt_path}: {e}")

    print(f"\nSuccessfully rendered {len(rendered_images)} meshes")

    # Create grid if requested
    if create_grid and rendered_images:
        grid_path = os.path.join(
            output_dir, f"overfitted_mlp_mesh_grid_{grid_rows}x{grid_cols}.png"
        )
        print(f"\nCreating grid image...")
        create_grid_image(
            rendered_images,
            grid_path,
            rows=grid_rows,
            cols=grid_cols,
            pad=grid_pad,
            row_gap=grid_row_gap,
            col_gap=grid_col_gap,
            square_tiles=False,
        )
        print(f"Grid saved to {grid_path}")

    return rendered_images


def parse_arguments():
    """Parses command-line arguments."""
    p = configargparse.ArgumentParser(
        description="Render 3D meshes from MLP checkpoints."
    )
    p.add_argument("-c", "--config_filepath", required=False, is_config_file=True)

    # Paths
    p.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to a single MLP checkpoint file or directory.",
    )
    p.add_argument(
        "--max_checkpoints",
        type=int,
        default=None,
        help="Maximum number of checkpoints to process.",
    )
    p.add_argument(
        "--mlp_config",
        type=str,
        required=True,
        help="Path to the MLP config YAML file describing architecture.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations/rendered_from_mlp",
        help="Output directory for generated meshes and renders.",
    )

    # Mesh generation parameters
    p.add_argument(
        "--resolution",
        type=int,
        default=800,
        help="Resolution for marching cubes mesh extraction.",
    )
    p.add_argument(
        "--output_type",
        type=str,
        default="occ",
        choices=["occ", "sdf", "logits"],
        help="Output type of the MLP model.",
    )
    # Grid parameters
    p.add_argument(
        "--create_grid",
        action="store_true",
        help="Create a grid image from all rendered meshes.",
    )

    return p.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Validate arguments
    if not args.checkpoint_path:
        print("Error: Specify --checkpoint_path (single file or directory)")
        sys.exit(1)

    skip_files = [
        "occ_106dfe858cb8fbc2afc6b80d80a265ab_model_final",
        "occ_10e4331c34d610dacc14f1e6f4f4f49b_model_final",
    ]

    # Check if checkpoint_path is a directory or file
    if os.path.isdir(args.checkpoint_path):
        # Batch processing
        print(f"\n{'='*60}")
        print("Batch rendering MLP checkpoints")
        print(f"{'='*60}")
        batch_render_checkpoints(
            args.checkpoint_path,
            args.mlp_config,
            args.output_dir,
            resolution=args.resolution,
            output_type=args.output_type,
            max_checkpoints=args.max_checkpoints,
            create_grid=args.create_grid,
            grid_rows=3,
            grid_cols=4,
            grid_pad=2,
            grid_row_gap=-10,
            grid_col_gap=4,
            skip_files=skip_files,
        )
    else:
        # Single checkpoint
        print(f"\n{'='*60}")
        print("Rendering single MLP checkpoint")
        print(f"{'='*60}")
        render_checkpoint_mesh(
            args.checkpoint_path,
            args.mlp_config,
            args.output_dir,
            resolution=args.resolution,
            output_type=args.output_type,
        )


if __name__ == "__main__":
    main()
