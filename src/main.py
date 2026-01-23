import os

import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.abspath("external"))

from datetime import datetime
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader

import wandb
from external.ldm.ldm.modules.diffusionmodules import openaimodel
from src.dataset import VoxelDataset, WeightDataset
from src.hd_utils import Config, get_mlp
from src.hyperdiffusion import HyperDiffusion
from src.transformer import Transformer


DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Setting PYOPENGL_PLATFORM based on device
if DEVICE.type == "cuda":
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    # os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    accelerator = "gpu"
elif DEVICE.type == "mps":
    accelerator = "cpu"
else:
    accelerator = "cpu"


@hydra.main(
    version_base=None,
    config_path="../configs/diffusion_configs",
    config_name="train_plane",
)
def main(cfg: DictConfig):
    Config.config = cfg
    method = Config.get("method")
    mlp_kwargs = None

    # In HyperDiffusion, we need to know the specifications of MLPs that are used for overfitting
    if "hyper" in method:
        mlp_kwargs = Config.config["mlp_config"]["params"]

    wandb.init(
        project="hyperdiffusion",
        dir=Config.config["tensorboard_log_dir"],
        settings=wandb.Settings(_disable_stats=False, _disable_meta=False),
        tags=[Config.get("mode")],
        mode="disabled" if Config.get("disable_wandb") else "online",
        config=dict(Config.config),
    )

    wandb_logger = WandbLogger()
    wandb_logger.log_text("config", ["config"], [[str(Config.config)]])
    print("wandb", wandb.run.name, wandb.run.id)

    train_dt = val_dt = test_dt = None

    # Although it says train, it includes all the shapes but we only extract training ones in WeightDataset
    mlps_folder_train = Config.get("mlps_folder_train")

    # Initialize Transformer for HyperDiffusion
    if "hyper" in method:
        mlp = get_mlp(mlp_kwargs)
        state_dict = mlp.state_dict()
        layers = []
        layer_names = []
        for l in state_dict:
            shape = state_dict[l].shape
            layers.append(np.prod(shape))
            layer_names.append(l)
        model = Transformer(
            layers, layer_names, **Config.config["transformer_config"]["params"]
        ).to(device=DEVICE)
    # Initialize UNet for Voxel baseline
    else:
        model = openaimodel.UNetModel(**Config.config["unet_config"]["params"]).float()

    dataset_path = os.path.join(Config.config["dataset_dir"], Config.config["dataset"])
    split_suffix = cfg.split_suffix if cfg.split_suffix else ""

    train_split_path = os.path.join(dataset_path, f"train_split{split_suffix}.lst")
    val_split_path = os.path.join(dataset_path, f"val_split{split_suffix}.lst")
    test_split_path = os.path.join(dataset_path, f"test_split{split_suffix}.lst")

    train_object_names = np.genfromtxt(train_split_path, dtype="str")
    if not cfg.mlp_config.params.move:
        train_object_names = set([str.split(".")[0] for str in train_object_names])

    # Check if dataset folder already has train,test,val split; create otherwise.
    if method == "hyper_3d":
        mlps_dataset_path = Config.get("mlps_folder_train")
        all_object_names = np.array(
            [obj for obj in os.listdir(mlps_dataset_path) if ".lst" not in obj]
        )
        total_size = len(all_object_names)
        val_size = int(total_size * 0.05)
        test_size = int(total_size * 0.15)
        train_size = total_size - val_size - test_size

        if not os.path.exists(train_split_path):
            train_idx = np.random.choice(
                total_size, train_size + val_size, replace=False
            )
            test_idx = set(range(total_size)).difference(train_idx)
            val_idx = set(np.random.choice(train_idx, val_size, replace=False))
            train_idx = set(train_idx).difference(val_idx)
            print(
                "Generating new partition",
                len(train_idx),
                train_size,
                len(val_idx),
                val_size,
                len(test_idx),
                test_size,
            )

            # Sanity checking the train, val and test splits
            assert len(train_idx.intersection(val_idx.intersection(test_idx))) == 0
            assert len(train_idx.union(val_idx.union(test_idx))) == total_size
            assert (
                len(train_idx) == train_size
                and len(val_idx) == val_size
                and len(test_idx) == test_size
            )

            np.savetxt(
                train_split_path,
                all_object_names[list(train_idx)],
                delimiter=" ",
                fmt="%s",
            )
            np.savetxt(
                val_split_path,
                all_object_names[list(val_idx)],
                delimiter=" ",
                fmt="%s",
            )
            np.savetxt(
                test_split_path,
                all_object_names[list(test_idx)],
                delimiter=" ",
                fmt="%s",
            )

        val_object_names = np.genfromtxt(val_split_path, dtype="str")
        val_object_names = set([str.split(".")[0] for str in val_object_names])
        test_object_names = np.genfromtxt(test_split_path, dtype="str")
        test_object_names = set([str.split(".")[0] for str in test_object_names])

        train_dt = WeightDataset(
            mlps_folder_train,
            wandb_logger,
            model.dims,
            mlp_kwargs,
            cfg,
            train_object_names,
        )
        train_dl = DataLoader(
            train_dt,
            batch_size=Config.get("batch_size"),
            shuffle=True,
            num_workers=4,
            pin_memory=False,
        )
        val_dt = WeightDataset(
            mlps_folder_train,
            wandb_logger,
            model.dims,
            mlp_kwargs,
            cfg,
            val_object_names,
        )
        test_dt = WeightDataset(
            mlps_folder_train,
            wandb_logger,
            model.dims,
            mlp_kwargs,
            cfg,
            test_object_names,
        )
    elif method == "raw_3d":
        dataset_path = os.path.join(
            Config.config["dataset_dir"], Config.config["dataset"]
        )
        train_dt = VoxelDataset(
            dataset_path, wandb_logger, model.dims, mlp_kwargs, cfg, train_object_names
        )
        train_dl = DataLoader(
            train_dt, batch_size=Config.get("batch_size"), shuffle=True, num_workers=2
        )

    # These two dl's are just placeholders, during val and test evaluation we are looking at test_split.lst,
    # val_split.lst files, inside calc_metrics methods
    val_dl = DataLoader(
        torch.utils.data.Subset(train_dt, [0]),
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )
    test_dl = DataLoader(
        torch.utils.data.Subset(train_dt, [0]),
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    print(
        "Train dataset length: {} Val dataset length: {} Test dataset length".format(
            len(train_dt), len(val_dt), len(test_dt)
        )
    )
    input_data = next(iter(train_dl))[0]
    print(
        "Input data shape, min, max:",
        input_data.shape,
        input_data.min(),
        input_data.max(),
    )

    best_model_save_path = Config.get("best_model_save_path")
    model_resume_path = Config.get("model_resume_path")

    # Initialize HyperDiffusion
    diffuser = HyperDiffusion(
        model, train_dt, val_dt, test_dt, mlp_kwargs, input_data.shape, method, cfg
    )

    # Specify where to save checkpoints
    checkpoint_path = os.path.join(
        Config.config["tensorboard_log_dir"],
        "lightning_checkpoints",
        f"{str(datetime.now()).replace(':', '-') + '-' + wandb.run.name + '-' + wandb.run.id}",
    )
    best_acc_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val/1-NN-CD-acc",
        mode="min",
        dirpath=checkpoint_path,
        filename="best-val-nn-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
    )

    best_mmd_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val/lgan_mmd-CD",
        mode="min",
        dirpath=checkpoint_path,
        filename="best-val-mmd-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
    )

    last_model_saver = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="last-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
        save_on_train_epoch_end=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=Config.get("epochs"),
        strategy=DDPPlugin(find_unused_parameters=False),
        logger=wandb_logger,
        default_root_dir=checkpoint_path,
        callbacks=[
            best_acc_checkpoint,
            best_mmd_checkpoint,
            last_model_saver,
            lr_monitor,
        ],
        check_val_every_n_epoch=Config.get("val_fid_calculation_period"),
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )

    if Config.get("mode") == "train":
        # If model_resume_path is provided (i.e., not None), the training will continue from that checkpoint
        trainer.fit(diffuser, train_dl, val_dl, ckpt_path=model_resume_path)

    # best_model_save_path is the path to saved best model
    trainer.test(
        diffuser,
        test_dl,
        ckpt_path=best_model_save_path if Config.get("mode") == "test" else None,
    )
    wandb_logger.finalize("Success")


if __name__ == "__main__":
    main()
