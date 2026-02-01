import os
import sys
import argparse
import colorsys
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from omegaconf import OmegaConf
import pathlib
from typing import Optional
from sklearn.cluster import AgglomerativeClustering

# Add the root of the project to the python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.dataset_utils import (
    load_semantic_point_cloud,
    load_meta_data,
    numeric_labels_to_str,
)

color_map = {
    "chair-back": "#1f77b4",
    "chair-seat": "#00b894",
    "chair-leg": "#f8d941",
    "chair-arm": "#9709e3",
    "plane-body": "#d63031",
    "plane-wing": "#eb9100",
    "plane-tail": "#085300",
    "plane-engine": "#041bc3",
}


def load_config(path):
    return OmegaConf.load(path)


def get_color_for_label(label: str, cluster_id: Optional[int] = None):
    base_color = color_map.get(label, "#cccccc")
    if cluster_id is None or cluster_id < 0:
        return base_color
    rgb_base = to_rgb(base_color)
    h, s, v = colorsys.rgb_to_hsv(*rgb_base)
    cluster_offset = (cluster_id % 7) / 7.0
    s_new = min(1.0, s * (0.7 + 0.6 * cluster_offset))
    v_new = min(1.0, v * (0.7 + 0.6 * cluster_offset))
    return colorsys.hsv_to_rgb(h, s_new, v_new)


def get_part_weights(model_state_dict, part_name):
    part_weights = []
    for key, value in model_state_dict.items():
        if key.startswith(f"parts.{part_name}."):
            part_weights.append(value.flatten())
    if not part_weights:
        return None
    return torch.cat(part_weights)


def rectangles_overlap(rect1, rect2):
    x_min1, x_max1, y_min1, y_max1 = rect1
    x_min2, x_max2, y_min2, y_max2 = rect2
    return not (
        x_max1 < x_min2 or x_max2 < x_min1 or y_max1 < y_min2 or y_max2 < y_min1
    )


def compute_cluster_assignments(tsne_results, n_clusters=5):
    print(
        f"Performing Agglomerative Clustering (linkage='single') with {n_clusters} clusters..."
    )
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="single")
    cluster_assignments = clustering.fit_predict(tsne_results)
    return cluster_assignments


def visualize_similar_parts_by_label(
    all_weights,
    all_labels,
    all_filenames,
    tsne_results,
    output_dir,
    num_samples=5,
    metadata=None,
    base_path=None,
    dataset="all",
    cluster_assignments=None,
    n_global_clusters=5,
    sampled_indices_map=None,  # Optional: Pass pre-selected indices for consistency
):
    """Visualize similar parts: pick samples per global cluster and render a matrix."""
    os.makedirs(output_dir, exist_ok=True)
    similar_parts_dict = {}

    for cluster_id in range(n_global_clusters):
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        # Use pre-selected indices if available (for consistency with t-SNE highlights)
        if sampled_indices_map and cluster_id in sampled_indices_map:
            sampled_idx = np.array(list(sampled_indices_map[cluster_id]))
        else:
            sampled_idx = (
                cluster_indices
                if len(cluster_indices) <= num_samples
                else np.random.choice(cluster_indices, num_samples, replace=False)
            )

        sampled_filenames = [all_filenames[i] for i in sampled_idx]
        cluster_ids = [cluster_id for _ in sampled_idx]
        labels = [all_labels[i] for i in sampled_idx]

        similar_parts_dict[cluster_id] = (sampled_filenames, cluster_ids, labels)
        print(f"Global Cluster {cluster_id} ({len(sampled_idx)} samples):")
        print(f"  Labels: {labels}")

    if not similar_parts_dict:
        print("No clusters found for visualization")
        return

    _visualize_similar_parts_as_pointclouds(
        similar_parts_dict, output_dir, metadata, base_path, dataset, num_samples
    )


def _visualize_similar_parts_as_pointclouds(
    similar_parts_dict, output_dir, metadata, base_path, dataset, num_samples
):
    """Visualize similar parts as actual 2D point cloud renderings (one row per global cluster)."""
    num_clusters = len(similar_parts_dict)

    # Setup figure with extra height/width to ensure labels fit
    fig, axes = plt.subplots(
        num_clusters,
        num_samples,
        figsize=(1.5 * num_samples, 1.0 * num_clusters),
        squeeze=False,
    )

    for row_idx, (cluster_id, (filenames, cluster_ids, labels)) in enumerate(
        sorted(similar_parts_dict.items())
    ):
        for col_idx, (filename, sample_cluster_id, label) in enumerate(
            zip(filenames, cluster_ids, labels)
        ):
            ax = axes[row_idx, col_idx]
            try:
                if label.startswith("chair-"):
                    category = "Chair"
                elif label.startswith("plane-"):
                    category = "Airplane"
                else:
                    category = label.split("-")[0].capitalize()

                if "occ_" in filename and "_model" in filename:
                    file_id = filename.split("occ_")[1].split("_model")[0]
                else:
                    file_id = filename.replace(".pt", "").replace(".pth", "")

                point_cloud, point_cloud_labels, label_names = (
                    load_semantic_point_cloud(
                        file_id=file_id,
                        base_path=base_path,
                        meta_data=metadata,
                        category=category,
                        expert_verified=False,
                    )
                )
                pc_labels = numeric_labels_to_str(point_cloud_labels, label_names)

                # filter point cloud by part label
                part_label = label.split("-")[1]
                part_indices = np.where(np.array(pc_labels) == part_label)[0]
                point_cloud = point_cloud[part_indices]

                # Use the original color for this label from the color_map
                point_color = color_map.get(label, "#cccccc")
                ax.scatter(
                    point_cloud[:, 0],
                    point_cloud[:, 1],
                    color=point_color,
                    s=4,
                )
                ax.set_title(f"{label}", fontsize=8)

                # Set consistent axis limits to avoid zooming in
                if len(point_cloud) > 0:
                    x_min, x_max = point_cloud[:, 0].min(), point_cloud[:, 0].max()
                    y_min, y_max = point_cloud[:, 1].min(), point_cloud[:, 1].max()
                    x_range = x_max - x_min if x_max > x_min else 1.0
                    y_range = y_max - y_min if y_max > y_min else 1.0
                    padding = 0.15
                    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
                    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

            except Exception as e:
                print(f"Could not load {filename}: {e}")
                ax.text(
                    0.5,
                    0.5,
                    "Error",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            ax.set_frame_on(False)

    # Use figure-level text for row labels to ensure alignment
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    sorted_clusters = sorted(similar_parts_dict.keys())

    for row_idx, cluster_id in enumerate(sorted_clusters):
        # Get position of the first axis in the row
        ax = axes[row_idx, 0]
        bbox = ax.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2

        fig.text(
            0.02,
            y_center,
            f"Cluster {cluster_id}",
            fontsize=10,
            rotation=90,
            va="center",
            ha="center",
        )

    output_path = os.path.join(output_dir, "similar_parts_matrix_pointclouds.svg")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved point cloud visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE visualization of MLP weight spaces."
    )
    parser.add_argument(
        "--dataset", type=str, default="all", choices=["all", "chair", "plane"]
    )
    parser.add_argument("--save_similar_parts", action="store_true")
    parser.add_argument("--n_clusters", type=int, default=5)
    args = parser.parse_args()

    # --- Configuration ---
    chair_config_path = "configs/overfitting_configs/overfit_chair_equal.yaml"
    plane_config_path = "configs/overfitting_configs/overfit_plane_equal.yaml"
    chair_mlp_dir = "mlp_weights/overfit_chair_vmap"
    plane_mlp_dir = "mlp_weights/overfit_plane_new_loss"
    output_plot_path = f"visualizations/weightspace_tsne_{args.dataset}.svg"

    chair_config = OmegaConf.load(chair_config_path)
    plane_config = OmegaConf.load(plane_config_path)
    chair_parts = list(chair_config.part_distribution.keys())
    plane_parts = list(plane_config.part_distribution.keys())

    all_weights = []
    all_labels = []
    all_filenames = []

    # Load Chairs
    if args.dataset in ["all", "chair"]:
        print(f"Processing chair MLPs...")
        if os.path.exists(chair_mlp_dir):
            for filename in os.listdir(chair_mlp_dir):
                if filename.endswith(".pt") or filename.endswith(".pth"):
                    filepath = os.path.join(chair_mlp_dir, filename)
                    state_dict = torch.load(filepath, map_location="cpu")
                    if "model_state_dict" in state_dict:
                        state_dict = state_dict["model_state_dict"]
                    for part_name in chair_parts:
                        weights = get_part_weights(state_dict, part_name)
                        if weights is not None:
                            all_weights.append(weights.numpy())
                            all_labels.append(f"chair-{part_name}")
                            all_filenames.append(filename)

    # Load Planes
    if args.dataset in ["all", "plane"]:
        print(f"Processing plane MLPs...")
        if os.path.exists(plane_mlp_dir):
            for filename in os.listdir(plane_mlp_dir):
                if filename.endswith(".pt") or filename.endswith(".pth"):
                    filepath = os.path.join(plane_mlp_dir, filename)
                    state_dict = torch.load(filepath, map_location="cpu")
                    if "model_state_dict" in state_dict:
                        state_dict = state_dict["model_state_dict"]
                    for part_name in plane_parts:
                        weights = get_part_weights(state_dict, part_name)
                        if weights is not None:
                            all_weights.append(weights.numpy())
                            all_labels.append(f"plane-{part_name}")
                            all_filenames.append(filename)

    if not all_weights:
        print("No weights were loaded. Exiting.")
        return

    print(f"Loaded {len(all_weights)} weight vectors.")

    # --- t-SNE ---
    print("Performing t-SNE...")
    weights_matrix = np.array(all_weights)
    scaler = StandardScaler()
    weights_matrix = scaler.fit_transform(weights_matrix)

    n_pca_components = min(50, len(weights_matrix), weights_matrix.shape[1])
    pca = PCA(n_components=n_pca_components)
    pca_result = pca.fit_transform(weights_matrix)

    tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=30,
        n_iter=1000,
        metric="cosine",
        init="pca",
        learning_rate="auto",
    )
    tsne_results = tsne.fit_transform(pca_result)

    # --- Clustering ---
    print("Computing cluster assignments...")
    cluster_assignments = compute_cluster_assignments(
        tsne_results, n_clusters=args.n_clusters
    )

    # --- Pre-computation for Boxes and Samples ---

    # 1. Compute Boxes
    global_cluster_boxes = {}
    global_cluster_sizes = {}
    for cluster_id in range(args.n_clusters):
        indices = np.where(cluster_assignments == cluster_id)[0]
        if len(indices) > 0:
            x_coords, y_coords = tsne_results[indices, 0], tsne_results[indices, 1]
            x_min, x_max, y_min, y_max = (
                np.min(x_coords),
                np.max(x_coords),
                np.min(y_coords),
                np.max(y_coords),
            )
            pad_x, pad_y = (x_max - x_min) * 0.1 or 0.5, (y_max - y_min) * 0.1 or 0.5
            global_cluster_boxes[cluster_id] = (
                x_min - pad_x,
                x_max + pad_x,
                y_min - pad_y,
                y_max + pad_y,
            )
            global_cluster_sizes[cluster_id] = len(indices)

    # 2. Filter Overlapping Boxes
    sorted_clusters_by_size = sorted(
        global_cluster_boxes.keys(),
        key=lambda cid: global_cluster_sizes[cid],
        reverse=True,
    )
    non_overlapping_clusters = set()
    for cluster_id in sorted_clusters_by_size:
        if not any(
            rectangles_overlap(
                global_cluster_boxes[cluster_id], global_cluster_boxes[sid]
            )
            for sid in non_overlapping_clusters
        ):
            non_overlapping_clusters.add(cluster_id)

    # 3. Pre-select samples (ensures matrix and plot highlights match)
    sampled_per_cluster = {}
    for cluster_id in range(args.n_clusters):
        indices = np.where(cluster_assignments == cluster_id)[0]
        if len(indices) == 0:
            continue
        sampled_idx = (
            np.random.choice(indices, 5, replace=False) if len(indices) > 5 else indices
        )
        sampled_per_cluster[cluster_id] = set(sampled_idx.tolist())

    # --- Plotting Phase ---
    print(f"Setting up plot...")
    fig, ax = plt.subplots(figsize=(16, 10))
    unique_labels = sorted(list(set(all_labels)))
    labels_in_legend = set()

    # === STEP 1: Base Scatter Plot (No boxes, No highlights) ===
    for label in unique_labels:
        indices = [j for j, l in enumerate(all_labels) if l == label]
        cluster_color = color_map.get(label, "#cccccc")
        lbl_text = label if label not in labels_in_legend else None
        labels_in_legend.add(label)

        ax.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            color=cluster_color,
            label=lbl_text,
            s=50,
            alpha=0.7,
        )

    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("Dimension 1", fontsize=16)
    ax.set_ylabel("Dimension 2", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True)

    # >>> SAVE RAW PLOT <<<
    raw_path = output_plot_path.replace(".svg", "_no_boxes.svg")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    fig.savefig(raw_path, bbox_inches="tight")
    print(f"Saved raw plot to: {raw_path}")

    # === STEP 2: Add Highlights and Boxes ===

    # Add Highlights (Black outline)
    for label in unique_labels:
        indices = [j for j, l in enumerate(all_labels) if l == label]
        cluster_color = color_map.get(label, "#cccccc")

        # Intersect current label indices with sampled indices
        highlight_indices = [
            idx
            for idx in indices
            if cluster_assignments[idx] in sampled_per_cluster
            and idx in sampled_per_cluster[cluster_assignments[idx]]
        ]

        if highlight_indices:
            ax.scatter(
                tsne_results[highlight_indices, 0],
                tsne_results[highlight_indices, 1],
                facecolors=cluster_color,
                edgecolors="black",
                linewidths=1.2,
                s=90,
                alpha=0.9,
                zorder=12,
            )

    # Add Boxes
    for cluster_id in non_overlapping_clusters:
        padded_x_min, padded_x_max, padded_y_min, padded_y_max = global_cluster_boxes[
            cluster_id
        ]

        # Get color of first item in cluster
        first_idx = np.where(cluster_assignments == cluster_id)[0][0]
        box_color = color_map.get(all_labels[first_idx], "#cccccc")
        rgb_tuple = to_rgb(box_color)

        rect = Rectangle(
            (padded_x_min, padded_y_min),
            padded_x_max - padded_x_min,
            padded_y_max - padded_y_min,
            linewidth=2,
            edgecolor=rgb_tuple,
            facecolor="none",
            zorder=10,
        )
        ax.add_patch(rect)
        ax.text(
            padded_x_min,
            padded_y_max,
            f"C{cluster_id}",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=rgb_tuple,
                edgecolor=rgb_tuple,
                alpha=0.8,
            ),
            color="white",
            zorder=11,
        )

    # >>> SAVE FINAL PLOT <<<
    fig.savefig(output_plot_path, bbox_inches="tight")
    print(f"Saved annotated plot to: {output_plot_path}")

    # --- Matrix Visualization ---
    if args.save_similar_parts:
        print("\nFinding and saving similar parts per label...")
        output_dir = os.path.join(
            os.path.dirname(output_plot_path), f"similar_parts_{args.dataset}"
        )

        metadata, base_path = None, None
        try:
            base_path = pathlib.Path("./data/shapenetpart/PartAnnotation")
            metadata = load_meta_data(base_path / "metadata.json")
        except Exception as e:
            print(f"Could not load ShapeNetPart metadata: {e}")

        visualize_similar_parts_by_label(
            all_weights,
            all_labels,
            all_filenames,
            tsne_results,
            output_dir,
            num_samples=5,
            metadata=metadata,
            base_path=base_path,
            dataset=args.dataset,
            cluster_assignments=cluster_assignments,
            n_global_clusters=5,
            sampled_indices_map=sampled_per_cluster,  # Pass specific samples for consistency
        )
        print(f"Similar parts saved to: {output_dir}")


if __name__ == "__main__":
    main()
