"""Retrieve the common files of the ShapeNet and ShapeNet Part datasets.

* Intersection of ShapeNet and ShapeNetPart datasets
* Additionally remove problematic shape files
"""

import sys
import os
import logging

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from common import file_utils
import pathlib


def get_file_ids_in_dir(directory: pathlib.Path) -> set[str]:
    """Get all file names in a given directory."""
    # can be either .pts or .npy files
    file_names = {f.name for f in directory.glob("*.*") if f.suffix in {".pth"}}
    file_ids = {str.split(name, "_")[1] for name in file_names}
    return file_ids


def get_common_shapenet_file_ids(
    hyperdiffusion_file_ids: set[str],
    semantic_hd_file_ids: set[str],
) -> set[str]:
    """Get the common ShapeNet files between ShapeNet and ShapeNetPart datasets."""
    common_file_ids = hyperdiffusion_file_ids.intersection(semantic_hd_file_ids)
    return common_file_ids


def compute_split_sizes(total_size, val_ratio=0.05, test_ratio=0.15):
    """Compute train, val, test split sizes."""
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    return train_size, val_size, test_size


def generate_and_save_splits(
    all_object_names,
    train_size,
    val_size,
    test_size,
    train_split_path,
    val_split_path,
    test_split_path,
    seed=None,
):
    """Generate train/val/test splits and save them to files."""
    if os.path.exists(train_split_path):
        logging.info("Split files already exist. Skipping generation.")
        return

    rng = np.random.default_rng(seed)
    total_size = len(all_object_names)

    # Sample train + val
    train_val_idx = rng.choice(total_size, train_size + val_size, replace=False)

    # Split val from train
    val_idx = rng.choice(train_val_idx, val_size, replace=False)
    train_idx = np.setdiff1d(train_val_idx, val_idx, assume_unique=True)
    test_idx = np.setdiff1d(np.arange(total_size), train_val_idx, assume_unique=True)

    logging.info(
        f"Generating new partition: "
        f"train={len(train_idx)}/{train_size}, "
        f"val={len(val_idx)}/{val_size}, "
        f"test={len(test_idx)}/{test_size}"
    )

    # Sanity checks
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(np.intersect1d(val_idx, test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == total_size

    # Save splits
    for path, idx in [
        (train_split_path, train_idx),
        (val_split_path, val_idx),
        (test_split_path, test_idx),
    ]:
        np.savetxt(path, all_object_names[idx], delimiter=" ", fmt="%s")
        logging.info(f"Saved split to: {path}")


if __name__ == "__main__":
    category = "plane"  # chair
    logging.basicConfig(level=logging.INFO)

    # path to mlp weights
    base = pathlib.Path("./mlp_weights")
    hyperdiff_path = base / "3d_128_plane_multires_4_manifoldplus_slower_no_clipgrad"
    semantic_hd_path = base / "overfit_plane_new_loss"

    # path to save splits
    plane_path = pathlib.Path("./data/baseline/02691156")

    # optional: limit the number of files to consider
    MAX_FILES = 400

    hyperdiffusion_file_ids = get_file_ids_in_dir(hyperdiff_path)
    semantic_hd_file_ids = get_file_ids_in_dir(semantic_hd_path)
    logging.info(f"Hyperdiffusion files: {len(hyperdiffusion_file_ids)}")
    logging.info(f"Semantic Hyperdiffusion files: {len(semantic_hd_file_ids)}")

    common_file_ids = get_common_shapenet_file_ids(
        hyperdiffusion_file_ids,
        semantic_hd_file_ids,
    )
    logging.info(f"Number of common files: {len(common_file_ids)}")

    # Save common files
    common_file_ids_save_path = base / f"./{category}_common_mlps.txt"
    with open(common_file_ids_save_path, "w") as f:
        f.write("\n".join(sorted(common_file_ids)))
    logging.info(f"Common file IDs saved to: {common_file_ids_save_path}")

    # Generate and save train, val, test splits
    all_object_names = np.array(sorted([f"{fid}.obj" for fid in common_file_ids]))
    if MAX_FILES is not None:
        all_object_names = all_object_names[:MAX_FILES]
        logging.info(f"Using only first {MAX_FILES} files for splits.")

    train_size, val_size, test_size = compute_split_sizes(len(all_object_names))

    # Save splits for both datasets
    suffix = "_400" if MAX_FILES is not None else ""
    train_split_path = base / f"train_split{suffix}.lst"
    val_split_path = base / f"val_split{suffix}.lst"
    test_split_path = base / f"test_split{suffix}.lst"

    generate_and_save_splits(
        all_object_names,
        train_size,
        val_size,
        test_size,
        str(train_split_path),
        str(val_split_path),
        str(test_split_path),
    )
