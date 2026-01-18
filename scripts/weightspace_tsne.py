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
    """
    Generates a color for a given label, optionally adjusting for cluster_id.
    Uses the predefined color_map for base colors.
    """
    base_color = color_map.get(label, "#cccccc")  # Default to grey if label not found
    if cluster_id is None or cluster_id < 0:
        return base_color

    # Convert base color to HSV to adjust for cluster
    rgb_base = to_rgb(base_color)
    h, s, v = colorsys.rgb_to_hsv(*rgb_base)

    # Vary saturation/value slightly based on cluster_id for distinction
    # This ensures clusters within the same label have similar but distinct colors
    cluster_offset = (cluster_id % 7) / 7.0  # Cycle through 7 variations
    s_new = min(1.0, s * (0.7 + 0.6 * cluster_offset))
    v_new = min(1.0, v * (0.7 + 0.6 * cluster_offset))

    return colorsys.hsv_to_rgb(h, s_new, v_new)


def get_part_weights(model_state_dict, part_name):
    """Extracts and flattens all weights for a specific part from a state dict."""
    part_weights = []
    for key, value in model_state_dict.items():
        if key.startswith(f"parts.{part_name}."):
            part_weights.append(value.flatten())
    if not part_weights:
        return None
    return torch.cat(part_weights)


def rectangles_overlap(rect1, rect2):
    """Check if two rectangles overlap. Each rect is (x_min, x_max, y_min, y_max)."""
    x_min1, x_max1, y_min1, y_max1 = rect1
    x_min2, x_max2, y_min2, y_max2 = rect2

    # Rectangles don't overlap if one is completely to the left/right or above/below the other
    return not (
        x_max1 < x_min2 or x_max2 < x_min1 or y_max1 < y_min2 or y_max2 < y_min1
    )


def compute_cluster_assignments(tsne_results, n_clusters=5):
    """Compute cluster assignments using Agglomerative Clustering (Single Linkage)."""
    print(
        f"Performing Agglomerative Clustering (linkage='single') with {n_clusters} clusters..."
    )

    # linkage='single' uses the minimum distance between observations of two sets.
    # This effectively captures non-convex, separated clusters (islands).
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
):
    """Visualize similar parts: pick samples per global cluster and render a matrix."""
    os.makedirs(output_dir, exist_ok=True)

    similar_parts_dict = {}

    for cluster_id in range(n_global_clusters):
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]

        if len(cluster_indices) == 0:
            print(f"Global Cluster {cluster_id}: no samples")
            continue

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
        print(f"  Files: {sampled_filenames}")
        print(f"  Cluster IDs: {cluster_ids}")

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

    fig, axes = plt.subplots(
        num_clusters,
        num_samples,
        figsize=(3 * num_samples, 1 * num_clusters),
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
                # Determine category from label
                if label.startswith("chair-"):
                    category = "Chair"
                elif label.startswith("plane-"):
                    category = "Airplane"
                else:
                    category = label.split("-")[0].capitalize()

                # Extract file_id from filename (format: occ_<id>_model_final.pth)
                if "occ_" in filename and "_model" in filename:
                    file_id = filename.split("occ_")[1].split("_model")[0]
                else:
                    # Fallback: remove .pt/.pth extension
                    file_id = filename.replace(".pt", "").replace(".pth", "")

                # Load point cloud
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

                # Prepare colors
                unique_labels = sorted(list(set(pc_labels)))
                label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
                colors = [label_to_id[lbl] for lbl in pc_labels]

                # Plot
                ax.scatter(
                    point_cloud[:, 0], point_cloud[:, 1], c=colors, s=2, cmap="viridis"
                )
                # Include semantic label and cluster ID in title
                if sample_cluster_id >= 0:
                    ax.set_title(f"{label}", fontsize=8)
                else:
                    ax.set_title(f"{label}", fontsize=8)

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

            # Add cluster ID label on first column
            if col_idx == 0:
                ax.set_ylabel(f"Cluster {cluster_id}", fontsize=10, fontweight="bold")

    # plt.suptitle(
    #     "Samples from t-SNE Global Clusters (5 samples per cluster)", fontsize=12
    # )
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, "similar_parts_matrix_pointclouds.svg")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved point cloud visualization to {output_path}")
    plt.close()


def main():
    """Main function to run the t-SNE visualization."""
    parser = argparse.ArgumentParser(
        description="t-SNE visualization of MLP weight spaces."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "chair", "plane"],
        help="Which dataset to process (all, chair, or plane)",
    )
    parser.add_argument(
        "--save_similar_parts",
        action="store_true",
        help="Save lists of similar parts per category based on t-SNE clustering",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of clusters per label (uses KMeans). If not specified, defaults to 5 clusters.",
    )
    args = parser.parse_args()

    # --- Configuration ---
    chair_config_path = "configs/overfitting_configs/overfit_chair_equal.yaml"
    plane_config_path = "configs/overfitting_configs/overfit_plane_equal.yaml"

    chair_mlp_dir = "mlp_weights/overfit_chair_vmap"
    plane_mlp_dir = "mlp_weights/overfit_plane_new_loss"

    output_plot_path = f"visualizations/weightspace_tsne_{args.dataset}.svg"

    # --- Load Configs and Define Parts ---

    chair_config = OmegaConf.load(chair_config_path)
    plane_config = OmegaConf.load(plane_config_path)

    chair_parts = list(chair_config.part_distribution.keys())
    plane_parts = list(plane_config.part_distribution.keys())

    all_weights = []
    all_labels = []
    all_filenames = []

    # --- Process Chair MLPs ---
    if args.dataset in ["all", "chair"]:
        print(f"Processing chair MLPs from: {chair_mlp_dir}")
        for filename in os.listdir(chair_mlp_dir):
            if filename.endswith(".pt") or filename.endswith(".pth"):
                filepath = os.path.join(chair_mlp_dir, filename)
                state_dict = torch.load(filepath, map_location="cpu")

                # If the model was saved with DataParallel or similar wrappers
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]

                for part_name in chair_parts:
                    weights = get_part_weights(state_dict, part_name)
                    if weights is not None:
                        all_weights.append(weights.numpy())
                        all_labels.append(f"chair-{part_name}")
                        all_filenames.append(filename)

    # --- Process Plane MLPs ---
    if args.dataset in ["all", "plane"]:
        print(f"Processing plane MLPs from: {plane_mlp_dir}")
        for filename in os.listdir(plane_mlp_dir):
            if filename.endswith(".pt") or filename.endswith(".pth"):
                filepath = os.path.join(plane_mlp_dir, filename)
                state_dict = torch.load(filepath, map_location="cpu")

                # If the model was saved with DataParallel or similar wrappers
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

    # --- Perform t-SNE ---

    print("Performing t-SNE...")
    weights_matrix = np.array(all_weights)

    # Normalize the weights
    scaler = StandardScaler()
    weights_matrix = scaler.fit_transform(weights_matrix)

    # PCA Reduction
    n_pca_components = min(50, len(weights_matrix), weights_matrix.shape[1])
    pca = PCA(n_components=n_pca_components)
    pca_result = pca.fit_transform(weights_matrix)

    print(f"PCA reduced shape: {pca_result.shape}")

    # t-SNE with Cosine metric and PCA initialization
    # Note: metric='cosine' is better for high-dim vectors
    tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=30,  # Lower perplexity might help separate distinct small clusters
        n_iter=1000,
        metric="cosine",
        init="pca",
        learning_rate="auto",
    )

    tsne_results = tsne.fit_transform(pca_result)

    # --- Compute Clusters ---
    print("Computing cluster assignments...")
    cluster_assignments = compute_cluster_assignments(
        tsne_results, n_clusters=args.n_clusters
    )

    # --- Plot Results ---

    # Matplotlib rendering
    print(f"Saving plot to: {output_plot_path}")
    fig, ax = plt.subplots(figsize=(16, 10))

    unique_labels = sorted(list(set(all_labels)))

    # First pass: compute global bounding boxes for all clusters
    global_cluster_boxes = {}
    global_cluster_sizes = {}
    for cluster_id in range(args.n_clusters):
        indices = np.where(cluster_assignments == cluster_id)[0]
        if len(indices) > 0:
            x_coords = tsne_results[indices, 0]
            y_coords = tsne_results[indices, 1]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)

            # Add padding to rectangle
            padding_x = (x_max - x_min) * 0.1 if x_max > x_min else 0.5
            padding_y = (y_max - y_min) * 0.1 if y_max > y_min else 0.5

            padded_x_min = x_min - padding_x
            padded_x_max = x_max + padding_x
            padded_y_min = y_min - padding_y
            padded_y_max = y_max + padding_y

            global_cluster_boxes[cluster_id] = (
                padded_x_min,
                padded_x_max,
                padded_y_min,
                padded_y_max,
            )
            global_cluster_sizes[cluster_id] = len(indices)

    # Second pass: select non-overlapping clusters using a greedy approach
    sorted_clusters_by_size = sorted(
        global_cluster_boxes.keys(),
        key=lambda cid: global_cluster_sizes[cid],
        reverse=True,
    )

    non_overlapping_clusters = set()
    for cluster_id in sorted_clusters_by_size:
        is_overlapping = False
        for selected_id in non_overlapping_clusters:
            if rectangles_overlap(
                global_cluster_boxes[cluster_id], global_cluster_boxes[selected_id]
            ):
                is_overlapping = True
                break
        if not is_overlapping:
            non_overlapping_clusters.add(cluster_id)

    # Track which labels have already been added to the legend
    labels_in_legend = set()

    # Third pass: plot points and boxes
    # Also precompute sampled indices per cluster for highlighting
    sampled_per_cluster = {}
    for cluster_id in range(args.n_clusters):
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        if len(cluster_indices) <= 5:
            sampled_idx = cluster_indices
        else:
            sampled_idx = np.random.choice(cluster_indices, 5, replace=False)
        sampled_per_cluster[cluster_id] = set(sampled_idx.tolist())

    for label in unique_labels:
        indices = [j for j, l in enumerate(all_labels) if l == label]

        # Use the original label-based color
        cluster_color = color_map.get(label, "#cccccc")

        # Only add label to legend if it hasn't been added yet
        if label not in labels_in_legend:
            cluster_label_text = label
            labels_in_legend.add(label)
        else:
            cluster_label_text = None

        ax.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            color=cluster_color,
            label=cluster_label_text,
            s=50,
            alpha=0.7,
        )

        # Highlight sampled points belonging to this label
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

    # Draw bounding boxes for non-overlapping global clusters
    for cluster_id in non_overlapping_clusters:
        padded_x_min, padded_x_max, padded_y_min, padded_y_max = global_cluster_boxes[
            cluster_id
        ]

        # Find a representative color from points in this cluster
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]
        if len(cluster_indices) > 0:
            # Use color from first label in this cluster
            first_label = all_labels[cluster_indices[0]]
            cluster_color = color_map.get(first_label, "#cccccc")
            rgb_tuple = to_rgb(cluster_color)

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

            # Add cluster ID text
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

    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("Dimension 1", fontsize=16)
    ax.set_ylabel("Dimension 2", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True)

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    fig.savefig(output_plot_path, bbox_inches="tight")
    print(f"Plot saved to: {output_plot_path}")

    # Save similar parts if requested
    if args.save_similar_parts:
        print("\nFinding and saving similar parts per label...")
        output_dir = os.path.join(
            os.path.dirname(output_plot_path), f"similar_parts_{args.dataset}"
        )

        # Load ShapeNetPart metadata if visualizing point clouds
        metadata = None
        base_path = None
        try:
            base_path = pathlib.Path("./data/shapenetpart/PartAnnotation")
            metadata_path = base_path / "metadata.json"
            metadata = load_meta_data(metadata_path)
            print("Loaded ShapeNetPart metadata for point cloud visualization")
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
        )
        print(f"Similar parts saved to: {output_dir}")


if __name__ == "__main__":
    main()
