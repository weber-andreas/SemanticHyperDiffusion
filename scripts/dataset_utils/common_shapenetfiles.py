"""Retrieve the common files of the ShapeNet and ShapeNet Part datasets.

* Intersection of ShapeNet and ShapeNetPart datasets
* Additionally remove problematic shape files
"""

import sys
import os
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.dataset_utils import load_meta_data
import pathlib


def get_file_ids_in_dir(directory: pathlib.Path) -> set[str]:
    """Get all file names in a given directory."""
    # can be either .pts or .npy files
    file_names = {
        f.name for f in directory.glob("*.*") if f.suffix in {".pts", ".npy", ".seg", ".obj"}
    }
    file_ids = {str.split(name, ".")[0] for name in file_names}
    return file_ids


def get_file_ids_from_txt(file_path: pathlib.Path) -> set[str]:
    """Get file IDs from a text file."""
    with open(file_path, "r") as f:
        file_ids = {line.strip() for line in f if line.strip()}
    return file_ids


def get_common_shapenet_file_ids(
    shapenetpart_file_ids: set[str],
    shapenet_file_ids: set[str],
    shapenetpart_expert_label_ids: set[str],
    problematic_file_ids: set[str],
) -> set[str]:
    """Get the common ShapeNet files between ShapeNet and ShapeNetPart datasets."""
    intersection = shapenetpart_file_ids.intersection(shapenet_file_ids)
    intersection = intersection.intersection(shapenetpart_expert_label_ids)
    common_file_ids = intersection.difference(problematic_file_ids)
    return common_file_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    CATEGORY = "Car"  # Airplane, Chair, Car

    shapenetpart_base_path = pathlib.Path("./data/shapenetpart/PartAnnotation")
    shapenet_base_path = pathlib.Path("./data/baseline/")

    meta_data_path = shapenetpart_base_path / "metadata.json"
    meta_data = load_meta_data(meta_data_path)

    # files path
    shapenetpart_points_path = (
        shapenetpart_base_path / meta_data[CATEGORY]["directory"] / "points"
    )
    shapenet_points_path = shapenet_base_path / (
        meta_data[CATEGORY]["directory"] #+ "_2048_pc"
    )
    shapenetpart_expert_label_path = (
        shapenetpart_base_path
        / meta_data[CATEGORY]["directory"]
        / "expert_verified/points_label"
    )

    shapenetpart_file_ids = get_file_ids_in_dir(shapenetpart_points_path)
    shapenet_file_ids = get_file_ids_in_dir(shapenet_points_path)
    shapenetpart_expert_label_ids = get_file_ids_in_dir(shapenetpart_expert_label_path)
    logging.info(f"ShapeNetPart files: {len(shapenetpart_file_ids)}")
    logging.info(f"ShapeNet files: {len(shapenet_file_ids)}")

    # get problematic files to exclude
    category_prefix = "plane" if CATEGORY == "Airplane" else CATEGORY.lower()
    problematic_file_path = (
        shapenet_base_path / f"{category_prefix}_problematic_shapes.txt"
    )
    problematic_file_ids = get_file_ids_from_txt(problematic_file_path)
    logging.info(f"Problematic files: {len(problematic_file_ids)}")

    common_file_ids = get_common_shapenet_file_ids(
        shapenetpart_file_ids,
        shapenet_file_ids,
        shapenetpart_expert_label_ids,
        problematic_file_ids,
    )
    logging.info(f"Common files: {len(common_file_ids)}")

    common_file_ids_save_path = pathlib.Path(
        f"./data/common/{category_prefix}_common_shapes.txt"
    )
    with open(common_file_ids_save_path, "w") as f:
        f.write("\n".join(sorted(common_file_ids)))
    logging.info(f"Common file IDs saved to: {common_file_ids_save_path}")
