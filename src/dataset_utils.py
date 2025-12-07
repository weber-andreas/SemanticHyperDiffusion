import json
import logging
import os
import pathlib
from glob import glob
from typing import Optional

import numpy as np
import natsort


def get_file_id(point_cloud_file: str) -> str:
    """Extract file ID from point cloud file name."""
    file_id = os.path.basename(point_cloud_file.removesuffix(".pts"))
    logging.info("Extracted file id: %s", file_id)
    return file_id


def load_semantic_point_cloud(
    base_path: pathlib.Path,
    category: str,
    meta_data: dict,
    specific_index: Optional[int] = None,
    file_id: Optional[str] = None,
    expert_verified: Optional[bool] = False,
) -> tuple[np.ndarray, list[str]]:
    """Load point clouds and their labels from the dataset."""

    if (specific_index is None and file_id is None) or (
        specific_index is not None and file_id is not None
    ):
        raise ValueError("Must provide exactly one of: specific_index or file_id")

    points_path = base_path / meta_data[category]["directory"] / "points"
    labels_path = base_path / meta_data[category]["directory"] / "points_label"
    labels_path_expert = (
        base_path / meta_data[category]["directory"] / "expert_verified/points_label"
    )
    label_names = meta_data[category]["lables"]
    numeric_to_label_name = {i: label_name for i, label_name in enumerate(label_names)}

    point_cloud = None
    point_cloud_labels = {}

    points_files = glob(os.path.join(points_path, "*.pts"))
    points_files = natsort.natsorted(points_files)

    if file_id is not None:
        # Find index of the file in point_files matching the given file_id
        try:
            specific_index = next(
                (
                    i
                    for i, file in enumerate(points_files)
                    if get_file_id(file) == file_id
                )
            )
        except StopIteration:
            raise ValueError(f"File ID {file_id} not found in points files.")

    point_cloud = np.loadtxt(points_files[specific_index]).astype("float32")
    file_id = get_file_id(points_files[specific_index])
    logging.info("Loaded file: %s", points_files[specific_index])

    # load expert verified labels
    if expert_verified:
        point_label_file = os.path.join(labels_path_expert, file_id + ".seg")
        if not os.path.exists(point_label_file):
            logging.warning(
                "Label file %s does not exist",
                point_label_file,
            )

        point_labels = np.loadtxt(point_label_file).astype("int32")
        point_cloud_labels = {}
        for numeric_id, label_name in numeric_to_label_name.items():
            mask = (point_labels == numeric_id).astype(int)
            point_cloud_labels[label_name] = mask
        print("Expert Verified")

    # load point cloud labels
    else:
        for label_name in label_names:
            point_label_file = os.path.join(labels_path, label_name, file_id + ".seg")
            if not os.path.exists(point_label_file):
                logging.warning(
                    "Label file %s does not exist. Skipping label %s.",
                    point_label_file,
                    label_name,
                )
                continue

            point_labels = np.loadtxt(point_label_file).astype("int32")
            point_cloud_labels[label_name] = point_labels
            logging.info("Loaded file: %s", point_label_file)

        label_counts = ", ".join(
            f"{part}: {point_labels.size}"
            for part, point_labels in point_cloud_labels.items()
        )
        logging.info("Loaded point cloud with %d points.", len(point_cloud))
        logging.info("Loaded point cloud with labels: %s", label_counts)

        assert (
            len(set(el.size for el in point_cloud_labels.values())) == 1
        ), "Mismatch in number of points between point cloud and labels."

    return point_cloud, point_cloud_labels, label_names


def numeric_labels_to_str(
    point_cloud_labels: dict[str, np.ndarray], label_names: list[str]
) -> list[str]:
    num_point_labels_per_category = len(next(iter(point_cloud_labels.values())))
    point_cloud_label_map = ["none"] * num_point_labels_per_category
    for label in label_names:
        # skip labels that are not in the point cloud
        if label not in point_cloud_labels:
            continue
        for i, data in enumerate(point_cloud_labels[label]):
            point_cloud_label_map[i] = label if data == 1 else point_cloud_label_map[i]
    return point_cloud_label_map


def load_meta_data(path: pathlib.Path) -> dict:
    """Load metadata for ShapeNetPart dataset."""
    with open(path, encoding="utf-8") as json_file:
        data = json.load(json_file)
        logging.info("Metadata loaded successfully.")
    return data
