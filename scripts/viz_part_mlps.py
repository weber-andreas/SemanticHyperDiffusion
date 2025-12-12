import argparse
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from tqdm import tqdm

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.mlp_decomposition.mlp_composite import get_model
from external.siren.dataio import get_mgrid

def process_and_visualize_shape(
    mlp_path: str,
    output_png_path: str,
    resolution: int,
    threshold: float,
    max_batch: int,
    device: torch.device
):
    """
    Loads one MLP, generates the part shapes, and saves the visualization as a PNG.
    """
    print(f"\n--- Processing: {Path(mlp_path).stem} ---")
    
    model = get_model(output_type="occ")
    state_dict = torch.load(mlp_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    part_names = list(model.registry.keys())

    # Create a normalized grid to query the MLP
    mgrid = get_mgrid(resolution, dim=3).to(device) * 0.5
    part_points = {}

    with torch.no_grad():
        for part_name in part_names:
            points_for_this_part = []
            for i in range(0, mgrid.shape[0], max_batch):
                coords_batch = mgrid[i:i + max_batch, :]
                model_input_dict = {"coords": coords_batch.unsqueeze(0)}
                model_output = model(model_input_dict, part_name=part_name)
                part_sdf = model_output['model_out'].squeeze(0)
                occupancy_probs = torch.sigmoid(part_sdf)
                inside_mask = occupancy_probs.squeeze() > threshold
                if inside_mask.any():
                    points_for_this_part.append(coords_batch[inside_mask].cpu().numpy())
            
            if points_for_this_part:
                part_points[part_name] = np.concatenate(points_for_this_part, axis=0)
                print(f"    -> Found {part_points[part_name].shape[0]} points for '{part_name}'")

    if not part_points:
        print("    -> No occupied points found across any parts. Skipping visualization.")
        return

    fig = go.Figure()
    
    for part_name, points in part_points.items():
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=2, opacity=0.8),
            name=f"Pred: {part_name}"
        ))

    fig.update_layout(
        title=f'Part Predictions for {Path(mlp_path).stem}',
        legend_title_text='Predicted Parts',
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(range=[-0.5, 0.5]),
            yaxis=dict(range=[-0.5, 0.5]),
            zaxis=dict(range=[-0.5, 0.5]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.write_image(output_png_path, width=1000, height=1000)
    print(f"    -> Visualization saved to '{output_png_path}'")

def main():
    """Main execution function to parse arguments and iterate through files."""
    parser = argparse.ArgumentParser(description="Generate visualizations for trained MLPComposite models.")
    
    parser.add_argument('--mlp_dir', type=str, required=True, help='Directory containing the .pth files of the overfitted models.')
    parser.add_argument('--output_dir', type=str, default='./visualizations', help='Directory to save the output PNG images.')
    
    parser.add_argument('--resolution', type=int, default=128, help='Resolution for the sampling grid.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Occupancy threshold for visualization.')
    parser.add_argument('--limit', type=int, default=None, help='Optional: Limit the number of shapes to process for a quick test.')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    mlp_files = sorted([f for f in os.listdir(args.mlp_dir) if f.endswith('.pth')])
    
    if args.limit:
        mlp_files = mlp_files[:args.limit]

    print(f"Found {len(mlp_files)} model checkpoints to process.")

    for mlp_filename in tqdm(mlp_files, desc="Generating Visualizations"):
        try:
            shape_id = mlp_filename.split('_')[1]
        except IndexError:
            print(f"Could not parse shape ID from '{mlp_filename}'. Skipping.")
            continue

        mlp_path = os.path.join(args.mlp_dir, mlp_filename)
        output_png_path = os.path.join(args.output_dir, f"{shape_id}_visualization.png")
            
        try:
            process_and_visualize_shape(
                mlp_path=mlp_path,
                output_png_path=output_png_path,
                resolution=args.resolution,
                threshold=args.threshold,
                max_batch=64**3,
                device=device
            )
        except Exception as e:
            print(f"An error occurred while processing {shape_id}: {e}")
            continue

if __name__ == '__main__':
    main()