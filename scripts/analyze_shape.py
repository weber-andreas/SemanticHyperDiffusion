import os
import sys
import torch
import numpy as np
import plotly.graph_objects as go
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.mlp_decomposition.mlp_composite import get_model
from external.siren.dataio import get_mgrid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def render_semantic_parts(model, output_path, parts, resolution=128, threshold=0.5):
    """Queries model grid and renders semantic parts interactively."""
    print(f"Generating Semantic Plot @ {resolution}^3...")
    mgrid = get_mgrid(resolution, dim=3).to(DEVICE) * 0.5
    batch_size = 64**3
    points_dict = {p: [] for p in parts}

    with torch.no_grad():
        for i in tqdm(range(0, mgrid.shape[0], batch_size), desc="Scanning Grid"):
            batch = mgrid[i:i+batch_size]
            output = model({"coords": batch.unsqueeze(0)})['parts']
            
            for part in parts:
                probs = torch.sigmoid(output[part]["model_out"].squeeze())
                mask = probs > threshold
                if mask.any():
                    points_dict[part].append(batch[mask].cpu().numpy())

    # Plotting
    fig = go.Figure()
    has_data = False
    for part, data in points_dict.items():
        if not data: continue
        pts = np.concatenate(data, axis=0)
        # Downsample for browser performance if heavy
        if len(pts) > 50000: pts = pts[np.random.choice(len(pts), 50000, replace=False)]
        
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers', marker=dict(size=2, opacity=0.8),
            name=part
        ))
        has_data = True

    if has_data:
        fig.update_layout(
            title="Semantic Part Decomposition",
            scene=dict(
                xaxis_title='X', 
                yaxis_title='Y', 
                zaxis_title='Z',
                xaxis=dict(range=[-0.5, 0.5]),
                yaxis=dict(range=[-0.5, 0.5]),
                zaxis=dict(range=[-0.5, 0.5]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)
            ),
        )
        out_html = f"{output_path}.html"
        fig.write_html(out_html)
        print(f"Interactive plot saved to {out_html}")
    else:
        print("Warning: No points found (Check threshold or weights).")

def main():
    parser = argparse.ArgumentParser(description="Analyze Composite MLP: Semantic Viz")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument('--cfg', type=str, default="configs/overfitting_configs/overfit_plane_equal.yaml")
    parser.add_argument('--out', type=str, default="analysis_results/shape", help="Output filename prefix")
    parser.add_argument('--res', type=int, default=256, help="Resolution for mesh (plot uses res/2)")
    parser.add_argument('--threshold', type=float, default=0.5, help="Occupancy threshold")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    model = get_model(cfg, model_type="composite", output_type="occ")
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    render_semantic_parts(model, args.out, cfg.label_names, resolution=args.res//2, threshold=args.threshold)

if __name__ == "__main__":
    main()