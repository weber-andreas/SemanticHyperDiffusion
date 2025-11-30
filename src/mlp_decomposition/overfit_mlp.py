"""Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement."""

import os
import sys
import hydra
import torch
import wandb
from functools import partial
from typing import List, Dict, Optional, Any
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)
sys.path.append("external/")
sys.path.append("external/siren/")

from external.siren import loss_functions, sdf_meshing, training, utils
from src.mlp_decomposition.test_mlp import SDFDecoder
from src.dataset import SemanticPointCloud
from src.mlp_decomposition.mlp_composite import get_model

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

EXCLUDED_FILES = [
    "train_split.lst",
    "test_split.lst",
    "val_split.lst",
    "train_split_small.lst",
    "test_split_small.lst",
    "val_split_small.lst",
]


def init_wandb(cfg: DictConfig) -> None:
    """Initializes Weights and Biases logging."""
    wandb.init(
        project="hyperdiffusion_overfitting",
        dir=cfg.wandb_dir,
        config=dict(cfg),
        mode="disabled",
    )


def get_file_ids(cfg: DictConfig) -> List[str]:
    """Retrieves and filters the list of object files to process."""
    pointcloud_folder = SemanticPointCloud.get_pc_folder_name(cfg)

    # check if common file ids are provided
    if cfg.common_file_ids is not None:
        with open(cfg.common_file_ids, "r") as f:
            file_ids = f.read().splitlines()
        return file_ids

    # otherwise use all files in specified folder
    file_ids = [f for f in os.listdir(pointcloud_folder) if f not in EXCLUDED_FILES]
    return file_ids


def get_paths(cfg: DictConfig, file_id: str) -> Dict[str, str]:
    """Constructs necessary file paths for a specific object."""
    return {
        "file_id": file_id,
        "pointcloud": os.path.join(cfg.dataset_folder, file_id + ".obj"),
        "expert": os.path.join(cfg.dataset_expert_folder, file_id + ".pts"),
        "label": os.path.join(cfg.label_folder, file_id + ".seg"),
        "logging_root": os.path.join(cfg.logging_root, cfg.exp_name),
    }


def create_dataloader(cfg: DictConfig, paths: Dict[str, str]) -> DataLoader:
    """Creates the Dataset and DataLoader for a specific object."""
    sdf_dataset = SemanticPointCloud(
        on_surface_points=cfg.batch_size,
        pointcloud_path=paths["pointcloud"],
        pointcloud_expert_path=paths["expert"],
        label_path=paths["label"],
        output_type=cfg.output_type,
        cfg=cfg,
    )
    return DataLoader(
        sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0
    )


def get_loss_function(cfg: DictConfig) -> Any:
    """Selects the appropriate loss function based on config."""
    if cfg.output_type == "occ":
        fn = (
            loss_functions.occ_tanh
            if cfg.out_act == "tanh"
            else loss_functions.occ_sigmoid
        )
    else:
        fn = loss_functions.sdf
    return partial(fn, cfg=cfg)


def get_checkpoint_filename(cfg: DictConfig, file_id: str) -> str:
    """Generates the standardized checkpoint filename."""
    filename = f"{cfg.output_type}_{file_id}"
    return filename


def handle_bad_initialization(
    model: torch.nn.Module, dataloader: DataLoader, loss_fn: Any, checkpoint_path: str
) -> bool:
    """
    Checks if a pre-existing model has high loss (bad initialization).
    Returns True if the model is 'bad' (outlier) and processing should skip/stop.
    """
    if not os.path.exists(checkpoint_path):
        return False

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    with torch.no_grad():
        try:
            model_input, gt = next(iter(dataloader))
            model_input = {k: v.to(DEVICE) for k, v in model_input.items()}
            gt = {k: v.to(DEVICE) for k, v in gt.items()}

            model_output = model(model_input)
            loss = loss_fn(model_output, gt, model)

            if loss.get("occupancy", 0) > 0.5:
                print("Outlier detected based on loss:", loss)
                return True
        except StopIteration:
            pass

    return False


def initialize_model_weights(
    model: torch.nn.Module,
    cfg: DictConfig,
    checkpoint_path: str,
    reference_state_dict: Optional[Dict],
) -> None:
    """Loads weights into the model based on the selected strategy."""

    if cfg.strategy == "continue" and os.path.exists(checkpoint_path):
        # Continue training from existing checkpoint
        model.load_state_dict(torch.load(checkpoint_path))

    elif reference_state_dict is not None and cfg.strategy not in [
        "random",
        "first_weights_kl",
    ]:
        # Initialize from a shared reference (e.g., from the first object trained)
        print("Loading shared reference weights.")
        model.load_state_dict(reference_state_dict)


def visualize_mesh(cfg: DictConfig, checkpoint_path: str, filename: str) -> None:
    """Generates and saves a mesh from the trained implicit function."""
    output_dir = os.path.join(cfg.logging_root, f"{cfg.exp_name}_ply")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)
    sdf_decoder = SDFDecoder(checkpoint_path, device=DEVICE)
    sdf_meshing.create_mesh(sdf_decoder, output_path, N=256, level=0, device=DEVICE)


def update_weight_statistics(x_0s: List[torch.Tensor], model: torch.nn.Module) -> None:
    """Collects flattened weights and prints variance statistics."""
    state_dict = model.state_dict()
    weights = [state_dict[w].flatten().cpu() for w in state_dict]
    weights_flat = torch.hstack(weights)

    x_0s.append(weights_flat)

    if len(x_0s) > 1:
        tmp = torch.stack(x_0s)
        var = torch.var(tmp, dim=0)
        print(
            f"Shape: {var.shape} | Mean: {var.mean().item():.5f} | "
            f"Std: {var.std().item():.5f} | Min: {var.min().item():.5f} | "
            f"Max: {var.max().item():.5f}"
        )


def process_single_object(
    cfg: DictConfig, file_id: str, idx: int, reference_state_dict: Optional[Dict]
) -> Optional[torch.nn.Module]:
    """
    Orchestrates the training pipeline for a single 3D object.
    Returns the trained model if successful, else None.
    """
    paths = get_paths(cfg, file_id)

    if not os.path.exists(paths["label"]):
        print(f"Label {paths['label']} not found. Skipping.")
        return None

    if cfg.strategy == "save_pc":
        print("Strategy 'save_pc': Skipping training for", paths["file_id"])
        return None

    dataloader = create_dataloader(cfg, paths)

    # Setup Filenames and Checkpoints
    filename = get_checkpoint_filename(cfg, paths["file_id"])
    checkpoint_path = os.path.join(paths["logging_root"], f"{filename}_model_final.pth")

    if (
        os.path.exists(checkpoint_path)
        and cfg.strategy != "continue"
        and cfg.strategy != "remove_bad"
    ):
        print("Checkpoint exists. Skipping:", checkpoint_path)
        return None

    # Initialize Model and Loss
    model = get_model(output_type=cfg.output_type).to(DEVICE)
    loss_fn = get_loss_function(cfg)

    # Handle Strategies (Remove Bad / Init)
    if cfg.strategy == "remove_bad":
        is_bad = handle_bad_initialization(model, dataloader, loss_fn, checkpoint_path)
        if is_bad:
            return None

    initialize_model_weights(model, cfg, checkpoint_path, reference_state_dict)

    # Training
    training.train(
        model=model,
        train_dataloader=dataloader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        steps_til_summary=cfg.steps_til_summary,
        epochs_til_checkpoint=cfg.epochs_til_ckpt,
        model_dir=paths["logging_root"],
        loss_fn=loss_fn,
        summary_fn=utils.wandb_sdf_summary,
        double_precision=False,
        clip_grad=cfg.clip_grad,
        wandb=wandb,
        filename=filename,
        cfg=cfg,
    )

    # Visualization (Only for first 5)
    if idx < 5:
        visualize_mesh(cfg, checkpoint_path, filename)

    return model


@hydra.main(
    version_base=None,
    config_path="../../configs/overfitting_configs",
    config_name="overfit_plane",
)
def main(cfg: DictConfig):
    init_wandb(cfg)

    with open_dict(cfg):
        cfg.mlp_config.output_type = cfg.output_type

    # Global State Initialization
    first_state_dict = None
    if cfg.strategy == "same_init":
        print("Initializing shared reference weights...")
        first_state_dict = get_model(output_type=cfg.output_type).state_dict()

    file_ids = get_file_ids(cfg)
    x_0s = []  # To store weights for stats

    for i, file_id in enumerate(file_ids):

        # Updated call: directly assigns the model, no tuple unpacking
        trained_model = process_single_object(
            cfg, file_id, idx=i, reference_state_dict=first_state_dict
        )

        if trained_model is None:
            continue

        # Keep weights of first MLP for Global State Initialization
        if (
            i == 0
            and first_state_dict is None
            and cfg.strategy in ["first_weights", "first_weights_kl"]
            and not cfg.multi_process.enabled
        ):
            first_state_dict = trained_model.state_dict()
            print(f"Captured first weights. LR: {cfg.lr}")

        update_weight_statistics(x_0s, trained_model)


if __name__ == "__main__":
    main()
