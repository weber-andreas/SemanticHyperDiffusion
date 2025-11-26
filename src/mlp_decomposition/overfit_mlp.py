"""Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement."""

import os
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
from functools import partial
import wandb

# Enable import from parent package
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)
sys.path.append("external/")
sys.path.append("external/siren/")


from external.siren import loss_functions, sdf_meshing, training, utils
from external.siren.experiment_scripts.test_sdf import SDFDecoder
from src.mlp_models import MLP3D
from src.dataset import SemanticPointCloud
from src.mlp_decomposition.mlp_composite import (
    print_model,
    CompositePartNet,
    MLPBudgetAllocator,
)


DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def get_model(cfg: DictConfig):
    distribution = {"wing": 0.2, "body": 0.5, "tail": 0.15, "engine": 0.15}
    target_params = 4 + 128 * 3 + 1

    allocator = MLPBudgetAllocator(target_params, distribution)
    registry = allocator.generate_registry()
    model = CompositePartNet(registry)
    print_model(model)
    return model


"""
python src/mlp_decomposition/overfit_mlp.py 
"""


@hydra.main(
    version_base=None,
    config_path="../../configs/overfitting_configs",
    config_name="overfit_plane",
)
def main(cfg: DictConfig):
    wandb.init(
        project="hyperdiffusion_overfitting",
        dir=cfg.wandb_dir,
        config=dict(cfg),
        mode="online",
    )
    first_state_dict = None
    if cfg.strategy == "same_init":
        first_state_dict = get_model(cfg).state_dict()
    x_0s = []
    with open_dict(cfg):
        cfg.mlp_config.output_type = cfg.output_type
    curr_lr = cfg.lr
    logging_root_path = os.path.join(cfg.logging_root, cfg.exp_name)
    mesh_jitter = cfg.mesh_jitter
    multip_cfg = cfg.multi_process
    files = [
        file
        for file in os.listdir(cfg.dataset_folder)
        if file not in ["train_split.lst", "test_split.lst", "val_split.lst"]
    ]

    lengths = []
    names = []
    train_object_names = np.genfromtxt(
        os.path.join(cfg.dataset_folder, "train_split.lst"), dtype="str"
    )

    for i, file in enumerate(files):
        # We used to have mesh jittering for augmentation but not using it anymore
        for j in range(10 if mesh_jitter and i > 0 else 1):
            # Quick workaround to rename from obj to off
            # if file.endswith(".obj"):
            #     file = file[:-3] + "off"

            tmp_file = file[:-4] if file.endswith(".npy") else file
            file_id = tmp_file.removesuffix(".obj")
            pointcloud_path = os.path.join(cfg.dataset_folder, file_id + ".obj.npy")
            label_path = os.path.join(cfg.label_folder, file_id + ".seg")

            # remove .npy
            if not (tmp_file in train_object_names):
                print(f"File {tmp_file} not in train_split")
                continue

            if not os.path.exists(label_path):
                print(f"Label {label_path} not found")
                continue

            sdf_dataset = SemanticPointCloud(
                on_surface_points=cfg.batch_size,
                pointcloud_path=pointcloud_path,
                label_path=label_path,
                is_mesh=False,
                output_type=cfg.output_type,
                cfg=cfg,
            )
            dataloader = DataLoader(
                sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0
            )
            if cfg.strategy == "save_pc":
                continue
            elif cfg.strategy == "diagnose":
                lengths.append(len(sdf_dataset.coords))
                names.append(file)
                continue

            # Define the model.
            model = get_model(cfg).to(DEVICE)

            # Define the loss
            loss_fn = loss_functions.sdf
            if cfg.output_type == "occ":
                loss_fn = (
                    loss_functions.occ_tanh
                    if cfg.out_act == "tanh"
                    else loss_functions.occ_sigmoid
                )
            loss_fn = partial(loss_fn, cfg=cfg)
            summary_fn = utils.wandb_sdf_summary

            filename = f"{file_id}_jitter_{j}"
            filename = f"{cfg.output_type}_{filename}"
            checkpoint_path = os.path.join(
                logging_root_path, f"{filename}_model_final.pth"
            )
            if os.path.exists(checkpoint_path):
                print("Checkpoint exists:", checkpoint_path)
                continue
            if cfg.strategy == "remove_bad":
                model.load_state_dict(torch.load(checkpoint_path))
                model.eval()
                with torch.no_grad():
                    (model_input, gt) = next(iter(dataloader))
                    model_input = {
                        key: value.cuda() for key, value in model_input.items()
                    }
                    gt = {key: value.cuda() for key, value in gt.items()}
                    model_output = model(model_input)
                    loss = loss_fn(model_output, gt, model)
                if loss["occupancy"] > 0.5:
                    print("Outlier:", loss)
                continue
            if cfg.strategy == "continue":
                if not os.path.exists(checkpoint_path):
                    continue
                model.load_state_dict(torch.load(checkpoint_path))
            elif (
                first_state_dict is not None
                and cfg.strategy != "random"
                and cfg.strategy != "first_weights_kl"
            ):
                print("loaded")
                model.load_state_dict(first_state_dict)

            training.train(
                model=model,
                train_dataloader=dataloader,
                epochs=cfg.epochs,
                lr=curr_lr,
                steps_til_summary=cfg.steps_til_summary,
                epochs_til_checkpoint=cfg.epochs_til_ckpt,
                model_dir=logging_root_path,
                loss_fn=loss_fn,
                summary_fn=summary_fn,
                double_precision=False,
                clip_grad=cfg.clip_grad,
                wandb=wandb,
                filename=filename,
                cfg=cfg,
            )
            if (
                i == 0
                and first_state_dict is None
                and (
                    cfg.strategy == "first_weights"
                    or cfg.strategy == "first_weights_kl"
                )
                and not multip_cfg.enabled
            ):
                first_state_dict = model.state_dict()
                print(curr_lr)
            state_dict = model.state_dict()

            # Calculate statistics on the MLP
            weights = []
            for weight in state_dict:
                weights.append(state_dict[weight].flatten().cpu())
            weights = torch.hstack(weights)
            x_0s.append(weights)
            tmp = torch.stack(x_0s)
            var = torch.var(tmp, dim=0)
            print(
                var.shape,
                var.mean().item(),
                var.std().item(),
                var.min().item(),
                var.max().item(),
            )
            print(var.shape, torch.var(tmp))

            # For the first 5 data, outputting shapes
            if i < 5:
                sdf_decoder = SDFDecoder(
                    cfg.model_type,
                    checkpoint_path,
                    "nerf" if cfg.model_type == "nerf" else "mlp",
                    cfg,
                )
                os.makedirs(
                    os.path.join(cfg.logging_root, f"{cfg.exp_name}_ply"),
                    exist_ok=True,
                )
                sdf_meshing.create_mesh(
                    sdf_decoder,
                    os.path.join(
                        cfg.logging_root,
                        f"{cfg.exp_name}_ply",
                        filename,
                    ),
                    N=256,
                    level=(
                        0
                        if cfg.output_type == "occ" and cfg.out_act == "sigmoid"
                        else 0
                    ),
                )


if __name__ == "__main__":
    main()
