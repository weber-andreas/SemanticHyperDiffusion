"""Precompute the distribution of the semantic labels in the pointclouds."""

import os
import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
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


def get_paths_and_metadata(category):
    shapenetpart_base_path = pathlib.Path("data/shapenetpart/PartAnnotation")
    shapenet_base_path = pathlib.Path("data/baseline/")

    meta_data_path = shapenetpart_base_path / "metadata.json"
    meta_data = load_meta_data(meta_data_path)

    directory = meta_data[category]["directory"]
    label_names = meta_data[category]["lables"]
    print(f"Category: {category}")
    print(f"Label names: {label_names}")

    paths = {
        "shapenetpart_base": shapenetpart_base_path,
        "shapenet_base": shapenet_base_path,
        "shapenetpart_points": shapenetpart_base_path / directory / "points",
        "shapenet_points": shapenet_base_path / (directory + "_2048_pc"),
        "shapenetpart_expert_label": shapenetpart_base_path
        / directory
        / "expert_verified/points_label",
    }
    return paths, directory, label_names


def get_common_files(paths, category):
    shapenetpart_file_ids = get_file_ids_in_dir(paths["shapenetpart_points"])
    shapenet_file_ids = get_file_ids_in_dir(paths["shapenet_points"])
    shapenetpart_expert_label_ids = get_file_ids_in_dir(
        paths["shapenetpart_expert_label"]
    )

    category_prefix = "plane" if category == "Airplane" else category.lower()
    problematic_file_path = (
        paths["shapenet_base"] / f"{category_prefix}_problematic_shapes.txt"
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
    return common_file_ids


def compute_distribution(common_file_ids, paths, directory, label_names):
    label_counts = {label: 0 for label in label_names}
    total_points = 0

    for file_id in tqdm(common_file_ids, desc="Processing files"):
        pointcloud_path = paths["shapenet_base"] / directory / f"{file_id}.obj"
        pointcloud_expert_path = (
            paths["shapenetpart_base"] / directory / "points" / f"{file_id}.pts"
        )
        pointcloud_expert_label_path = (
            paths["shapenetpart_base"]
            / directory
            / f"expert_verified/points_label/{file_id}.seg"
        )

        try:
            dataset_semantic_pc = SemanticPointCloud(
                on_surface_points=2048,
                pointcloud_path=str(pointcloud_path),
                pointcloud_expert_path=str(pointcloud_expert_path),
                label_path=str(pointcloud_expert_label_path),
                output_type="occ",
                cfg=OmegaConf.create(
                    {
                        "n_points": 2048,
                        "strategy": "first_weights",
                        "vox_resolution": 32,
                        "dataset_folder": str(paths["shapenet_base"] / directory),
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

    label_percentages = {
        label_name: (count / total_points) for label_name, count in label_counts.items()
    }
    return label_percentages


def visualize_distribution(label_percentages, category, output_path):

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Create bar plot
    names = list(label_percentages.keys())
    values = [label_percentages[n] * 100 for n in names]

    # Create color mapping consistent with viz_shapenetpart.py
    sorted_names = sorted(names)
    label_to_id = {name: i for i, name in enumerate(sorted_names)}

    cmap = matplotlib.colormaps["viridis"]
    norm = mcolors.Normalize(vmin=0, vmax=len(sorted_names) - 1)
    bar_colors = [cmap(norm(label_to_id[n])) for n in names]

    ax = sns.barplot(x=names, y=values, palette=bar_colors, hue=names, legend=False)

    plt.title(f"Semantic Label Distribution - {category}", fontsize=16)
    plt.xlabel("Part Name", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.ylim(0, 100)

    # Add percentage labels on top of bars
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")


def main():
    CATEGORY = "Chair"  # Airplane, Chair, Car

    paths, directory, label_names = get_paths_and_metadata(CATEGORY)
    common_file_ids = get_common_files(paths, CATEGORY)
    label_percentages = compute_distribution(
        common_file_ids, paths, directory, label_names
    )

    print("\nLabel Distribution:")
    for label_name in label_names:
        percentage = label_percentages[label_name]
        print(f"{label_name}: {percentage * 100:.2f}%")

    # Save label counts to a file
    save_file = f"data/common/{CATEGORY}_label_distribution.json"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, "w") as f:
        json.dump(label_percentages, f)

    # Visualization
    plot_path = f"data/common/{CATEGORY}_label_distribution.png"
    visualize_distribution(label_percentages, CATEGORY, plot_path)


if __name__ == "__main__":
    main()
