"""Precompute the distribution of the semantic labels in the pointclouds."""

import os
import pathlib
import sys

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import json

from scripts.dataset_utils.common_shapenetfiles import (
    get_common_shapenet_file_ids,
    get_file_ids_from_txt,
    get_file_ids_in_dir,
)
from src.dataset import SemanticPointCloud
from src.dataset_utils import load_meta_data


def main():
    CATEGORY = "Airplane"  # Airplane, Chair, Car

    # Paths
    shapenetpart_base_path = pathlib.Path("data/shapenetpart/PartAnnotation")
    shapenet_base_path = pathlib.Path("data/baseline/")

    meta_data_path = shapenetpart_base_path / "metadata.json"
    meta_data = load_meta_data(meta_data_path)

    directory = meta_data[CATEGORY]["directory"]
    label_names = meta_data[CATEGORY]["lables"]
    print(f"Category: {CATEGORY}")
    print(f"Label names: {label_names}")

    # File paths
    shapenetpart_points_path = shapenetpart_base_path / directory / "points"
    shapenet_points_path = shapenet_base_path / (directory + "_2048_pc")
    shapenetpart_expert_label_path = (
        shapenetpart_base_path / directory / "expert_verified/points_label"
    )

    # Get common files
    shapenetpart_file_ids = get_file_ids_in_dir(shapenetpart_points_path)
    shapenet_file_ids = get_file_ids_in_dir(shapenet_points_path)
    shapenetpart_expert_label_ids = get_file_ids_in_dir(shapenetpart_expert_label_path)

    category_prefix = "plane" if CATEGORY == "Airplane" else CATEGORY.lower()
    problematic_file_path = (
        shapenet_base_path / f"{category_prefix}_problematic_shapes.txt"
    )
    problematic_file_ids = get_file_ids_from_txt(problematic_file_path)

    common_file_ids = get_common_shapenet_file_ids(
        shapenetpart_file_ids,
        shapenet_file_ids,
        shapenetpart_expert_label_ids,
        problematic_file_ids,
    )

    common_file_ids = sorted(list(common_file_ids))
    print(f"Found {len(common_file_ids)} common files.")

    # Accumulators for label counts
    label_counts = {label: 0 for label in label_names}
    total_points = 0

    # Iterate over files
    for file_id in tqdm(common_file_ids, desc="Processing files"):
        pointcloud_path = shapenet_base_path / directory / f"{file_id}.obj"
        pointcloud_expert_path = (
            shapenetpart_base_path / directory / "points" / f"{file_id}.pts"
        )
        pointcloud_expert_label_path = (
            shapenetpart_base_path
            / directory
            / f"expert_verified/points_label/{file_id}.seg"
        )

        try:
            dataset_semantic_pc = SemanticPointCloud(
                on_surface_points=2048,
                pointcloud_path=str(pointcloud_path),
                pointcloud_expert_path=str(pointcloud_expert_path),
                label_path=str(pointcloud_expert_label_path),
                is_mesh=True,
                output_type="occ",
                cfg=OmegaConf.create(
                    {
                        "n_points": 2048,
                        "strategy": "first_weights",
                        "vox_resolution": 32,
                    }
                ),
            )

            labels = dataset_semantic_pc.labels

            for i, label_name in enumerate(label_names):
                count = np.sum(labels == (i + 1))
                label_counts[label_name] += count
                total_points += count

        except Exception as e:
            print(f"Error processing {file_id}: {e}")
            continue

    print("\nLabel Distribution:")
    label_percentages = {
        label_name: (count / total_points) for label_name, count in label_counts.items()
    }
    for label_name in label_names:
        percentage = label_percentages[label_name]
        print(f"{label_name *100}: {percentage:.2f}%")

    # Save label counts to a file
    save_file = f"data/common/{CATEGORY}_label_distribution.json"
    with open(save_file, "w") as f:
        json.dump(label_percentages, f)


if __name__ == "__main__":
    main()
