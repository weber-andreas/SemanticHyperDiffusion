"""Visualization of ShapeNetPart dataset"""

import sys
import os
import logging
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.abspath(ROOT_DIR))
from src.dataset_utils import (
    load_semantic_point_cloud,
    load_meta_data,
    numeric_labels_to_str,
)


def visualize_pointcloud_3d(point_cloud: np.ndarray, labels: list[str]) -> None:
    """Visualize point cloud data interactively using Plotly."""
    df = pd.DataFrame(
        data={
            "x": point_cloud[:, 0],
            "y": point_cloud[:, 1],
            "z": point_cloud[:, 2],
            "label": labels,
        }
    )
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="label",
        labels={"label": "Classes"},
        opacity=0.7,
    )

    fig.update_traces(
        marker=dict(size=5, line=dict(width=1, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        title="3D Point Cloud Visualization",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        legend_title="Labels",
        scene_aspectmode="data",
    )
    fig.show()


def visualize_single_pointcloud(
    metadata: dict,
    base_path: pathlib.Path,
    category: str = "Airplane",
    visualize_2d: bool = False,
    object_index: Optional[int] = None,
    file_id: Optional[str] = None,
) -> None:
    """Visualize a single point cloud with its labels."""
    if file_id:
        attr_filter = {"file_id": file_id}
    elif object_index:
        attr_filter = {"specific_index": object_index}
    else:
        raise ValueError("File ID or object index must be provided.")

    # Object categories
    object_categories = set(metadata.keys())
    logging.info("Object categories found: %s", object_categories)
    assert category in object_categories, f"Category {category} not found in metadata."

    point_cloud, point_cloud_labels, label_names = load_semantic_point_cloud(
        **attr_filter,  # either file_id or specific_index
        base_path=base_path,
        category=category,
        meta_data=metadata,
        expert_verified=args.expert_verified,
    )
    labels = numeric_labels_to_str(point_cloud_labels, label_names)

    if visualize_2d:
        visualize_pointcloud_2d(point_cloud, labels)
    else:
        visualize_pointcloud_3d(point_cloud, labels)


def visualize_pointcloud_2d(point_cloud: np.ndarray, labels: list[str]) -> None:
    """Visualize point cloud data using Matplotlib in 2D."""
    if point_cloud.shape[1] < 2:
        raise ValueError("Input point cloud must have at least 2 dimensions.")

    unique_labels = sorted(list(set(labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    colors = [label_to_id[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        point_cloud[:, 0], point_cloud[:, 1], c=colors, s=5, cmap="viridis"
    )

    # Create a legend
    legend_handles = []
    for label, i in label_to_id.items():
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=scatter.cmap(scatter.norm(i)),
                markersize=10,
                label=label,
            )
        )
    ax.legend(handles=legend_handles, title="Classes")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    plt.show()


def visualize_category_matrix(
    categories: list[str],
    base_path: pathlib.Path,
    meta_data: dict,
    num_objects: int = 5,
) -> None:
    """Visualize a matrix of 2D point cloud projections for different categories."""
    num_categories = len(categories)
    fig, axes = plt.subplots(
        num_categories,
        num_objects,
        figsize=(3 * num_objects, 3 * num_categories),
        squeeze=False,
    )

    for i, category in enumerate(categories):
        j = 0
        object_counts = 0
        while object_counts < num_objects:
            ax = axes[i, j]
            try:
                point_cloud, point_cloud_labels, label_names = (
                    load_semantic_point_cloud(
                        base_path=base_path,
                        meta_data=meta_data,
                        category=category,
                        specific_index=j,
                        expert_verified=args.expert_verified,
                    )
                )
                labels = numeric_labels_to_str(labels, label_names)

                unique_labels = sorted(list(set(labels)))
                label_to_id = {label: i for i, label in enumerate(unique_labels)}
                colors = [label_to_id[label] for label in labels]

                ax.scatter(
                    point_cloud[:, 0], point_cloud[:, 1], c=colors, s=2, cmap="viridis"
                )

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal", adjustable="box")
                ax.set_frame_on(False)  # Disable the box around the subplot
                # if j == 0:
                #     ax.set_ylabel(category, fontsize=14, rotation=90, labelpad=20)

                object_counts += 1

            except (FileNotFoundError, StopIteration) as e:
                logging.warning(
                    "Could not load object %d for category %s: %s", j, category, e
                )
                continue

            j += 1
    plt.tight_layout()
    plt.savefig(base_path / "category_matrix.png", dpi=600, bbox_inches="tight")
    plt.show()


def main(args: argparse.Namespace) -> None:
    base_path = pathlib.Path("./data/shapenetpart/PartAnnotation")
    metadata_path = base_path / "metadata.json"
    metadata = load_meta_data(metadata_path)

    if args.single_object:
        if len(args.categories) > 1:
            raise ValueError(
                "Only one category can be specified for single object visualization."
            )
        file_id = None if args.object_index is not None else args.file_id
        visualize_single_pointcloud(
            metadata=metadata,
            base_path=base_path,
            object_index=args.object_index,
            file_id=file_id,
            category=args.categories[0],
            visualize_2d=args.visualize_2d,
        )
    elif args.category_matrix:
        visualize_category_matrix(
            categories=args.categories,
            base_path=base_path,
            meta_data=metadata,
            num_objects=7,
        )
    else:
        raise ValueError("Invalid arguments.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    # Create a mutually exclusive group for the main modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--single_object",
        action="store_true",
        help="Visualize a single object.",
    )
    mode_group.add_argument(
        "--category_matrix",
        action="store_true",
        help="Visualize a matrix of different objects and categories.",
    )

    # Arguments for single object mode
    parser.add_argument(
        "--object_index",
        type=int,
        default=3,
        help="Index of object to visualize.",
    )
    parser.add_argument(
        "--file_id",
        type=str,
        # default="1a888c2c86248bbcf2b0736dd4d8afe0",
        help="File ID of object to visualize.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="Airplane",
        help="List of categories to visualize, e.g. 'Airplane' 'Chair'",
    )
    parser.add_argument(
        "--visualize_2d",
        action="store_true",
        default=False,
        help="Visualize point cloud in 2D.",
    )
    parser.add_argument(
        "--expert_verified",
        action="store_true",
        default=False,
        help="Use expert verified labels.",
    )
    args = parser.parse_args()
    args.categories = args.categories.split(" ")

    main(args)
