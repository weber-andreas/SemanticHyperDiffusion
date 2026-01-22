import argparse
import copy
import os
import sys
import torch
from omegaconf import OmegaConf, DictConfig

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.mlp_decomposition.mlp_composite import get_model as get_composite_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_mlp(cfg: DictConfig, weights_path: str):
    """Initializes model from config and loads weights."""
    model = get_composite_model(cfg, model_type="composite", output_type=cfg.output_type)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

def stitch_parts(base_model, donor_model, parts_to_transfer: list):
    """Returns a new model based on base_model, overwriting specific parts from donor_model."""
    hybrid_model = copy.deepcopy(base_model)
    
    print(f"Stitching: Base <- Donor for parts: {parts_to_transfer}")
    
    for part_name in parts_to_transfer:
        if part_name not in hybrid_model.parts:
            print(f"Warning: Part '{part_name}' not found in registry. Skipping.")
            continue
            
        # Transfer weights for this specific sub-network
        hybrid_model.parts[part_name].load_state_dict(
            donor_model.parts[part_name].state_dict()
        )
        
    return hybrid_model

def main():
    parser = argparse.ArgumentParser(description="Naively stitch semantic parts between two MLPComposite models.")
    
    # Defaults
    default_a = "mlp_weights/overfit_plane_new_loss/occ_1a04e3eab45ca15dd86060f189eb133_model_final.pth"
    default_b = "mlp_weights/overfit_plane_new_loss/occ_1a32f10b20170883663e90eaf6b4ca52_model_final.pth"
    
    parser.add_argument("--config", type=str, default="configs/overfitting_configs/overfit_plane_equal.yaml", help="Path to model config")
    parser.add_argument("--donor_a", type=str, default=default_a, help="Path to Donor A weights (Source of parts)")
    parser.add_argument("--donor_b", type=str, default=default_b, help="Path to Donor B weights (Base model)")
    parser.add_argument("--parts", nargs='+', default=['body', 'tail'], help="List of parts to take from Donor A")
    parser.add_argument("--out_dir", type=str, default="mlp_weights/stiched_results", help="Output directory")
    parser.add_argument("--out_name", type=str, default="naive_stitch.pth", help="Output filename")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    save_path = os.path.join(args.out_dir, args.out_name)

    print(f"Loading Donor A: {os.path.basename(args.donor_a)}")
    model_a = load_mlp(cfg, args.donor_a)
    
    print(f"Loading Donor B: {os.path.basename(args.donor_b)}")
    model_b = load_mlp(cfg, args.donor_b)

    hybrid_model = stitch_parts(model_b, model_a, args.parts)

    torch.save(hybrid_model.state_dict(), save_path)
    print(f"Saved stiched model to: {save_path}")

if __name__ == "__main__":
    main()