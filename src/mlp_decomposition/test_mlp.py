"""Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian."""

# Enable import from parent package
import os
import sys
from pathlib import Path
import torch
import configargparse
from omegaconf import OmegaConf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)
sys.path.append("external/")
sys.path.append("external/siren/")

from external.siren import sdf_meshing, utils
from src.mlp_decomposition.mlp_composite import get_model


class SDFDecoder(torch.nn.Module):
    def __init__(self, checkpoint_path, device, output_type="sdf", cfg=None):
        super().__init__()
        self.model = get_model(cfg, output_type=output_type)

        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(device)

    def forward(self, coords):
        model_in = {"coords": coords}
        return self.model(model_in)["model_out"]


def main():

    p = configargparse.ArgumentParser()
    p.add(
        "-c",
        "--config_filepath",
        required=False,
        is_config_file=True,
        help="Path to config file.",
    )
    p.add_argument(
        "--logging_root",
        type=str,
        default="./logs",
        help="root for logging",
    )
    p.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of subdirectory in logging_root where summaries and checkpoints will be saved.",
    )

    # General training options
    p.add_argument("--batch_size", type=int, default=16384)
    p.add_argument(
        "--checkpoint_path",
        default=None,
        help="Checkpoint to trained model.",
    )

    p.add_argument(
        "--model_type",
        type=str,
        default="sine",
        help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)',
    )
    p.add_argument(
        "--mode",
        type=str,
        default="mlp",
        help='Options are "mlp" or "nerf"',
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=1600,
    )
    p.add_argument(
        "--level",
        type=float,
        default=0.0,
        help="Isosurface level",
    )
    p.add_argument(
        "--output_type",
        type=str,
        default="occ",
        help="Output type (occ or sdf)",
    )

    opt = p.parse_args()

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if opt.config_filepath:
        cfg = OmegaConf.load(opt.config_filepath)
    else:
        cfg = None

    sdf_decoder = SDFDecoder(opt.checkpoint_path, DEVICE, output_type=opt.output_type, cfg=cfg)
    name = Path(opt.checkpoint_path).stem
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    utils.cond_mkdir(root_path)

    # Debug: Print model output statistics
    print("Analyzing model output range...")
    with torch.no_grad():
        # Create a small grid to test
        N_debug = 32
        voxel_origin = [-0.5] * 3
        voxel_size = 1.0 / (N_debug - 1)
        overall_index = torch.arange(0, N_debug**3, 1, out=torch.LongTensor())
        samples = torch.zeros(N_debug**3, 3)
        samples[:, 2] = overall_index % N_debug
        samples[:, 1] = (overall_index.long() / N_debug) % N_debug
        samples[:, 0] = ((overall_index.long() / N_debug) / N_debug) % N_debug
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        samples = samples.to(DEVICE)
        output = sdf_decoder.model(samples)["model_out"]
        print(f"Model Output Stats:")
        print(f"  Min: {output.min().item()}")
        print(f"  Max: {output.max().item()}")
        print(f"  Mean: {output.mean().item()}")
        print(f"  Std: {output.std().item()}")

    sdf_meshing.create_mesh(
        sdf_decoder,
        os.path.join(root_path, name),
        N=opt.resolution,
        level=opt.level,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
