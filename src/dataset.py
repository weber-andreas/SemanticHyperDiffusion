import os
import sys
from math import ceil, floor
from os.path import join

import numpy as np
import torch
import trimesh
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from trimesh.voxel import creation as vox_creation

# Enable import from parent package
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from external.siren.dataio import anime_read
from src.augment import random_permute_flat, random_permute_mlp, sorted_permute_mlp
from src.hd_utils import generate_mlp_from_weights, get_mlp
from scripts.dataset_utils.viz_shapenetpart import visualize_pointcloud_3d
from src.dataset_utils import load_meta_data


class VoxelDataset(Dataset):
    """Load 3d mesh data and convert it to voxel grids"""

    def __init__(
        self, mesh_folder, wandb_logger, model_dims, mlp_kwargs, cfg, object_names=None
    ):
        self.mesh_folder = mesh_folder
        if cfg.filter_bad:
            blacklist = set(np.genfromtxt(cfg.filter_bad_path, dtype=str))

        self.mesh_files = []
        if object_names is None:
            self.mesh_files = [
                file
                for file in list(os.listdir(mesh_folder))
                if file not in ["train_split.lst", "test_split.lst", "val_split.lst"]
            ]
        else:
            for file in list(os.listdir(mesh_folder)):
                if file.split(".")[0] in blacklist and cfg.filter_bad:
                    continue

                if (
                    ("_" in file and file.split("_")[1] in object_names)
                    or file in object_names
                    or file.split(".")[0] in object_names
                ):
                    self.mesh_files.append(file)
        self.transform = None
        self.logger = wandb_logger
        self.model_dims = model_dims
        self.cfg = cfg
        self.vox_folder = self.mesh_folder + "_vox"
        os.makedirs(self.vox_folder, exist_ok=True)

    def __getitem__(self, index):
        dir = self.mesh_files[index]
        path = join(self.mesh_folder, dir)
        resolution = self.cfg.vox_resolution
        voxel_size = 1.9 / (resolution - 1)
        total_time = self.cfg.unet_config.params.image_size

        # Animated objects
        if self.cfg.mlp_config.params.move:
            folder_name = os.path.basename(path)
            anime_file_path = os.path.join(path, folder_name + ".anime")
            nf, nv, nt, vert_data, face_data, offset_data = anime_read(anime_file_path)

            def normalize(obj, v_min, v_max):
                vertices = obj.vertices
                vertices -= np.mean(vertices, axis=0, keepdims=True)
                vertices *= 0.95 / (max(abs(v_min), abs(v_max)))
                obj.vertices = vertices
                return obj

            # total_time = min(nf, total_time)
            vert_datas = []
            v_min, v_max = float("inf"), float("-inf")

            frames = np.linspace(0, nf, total_time, dtype=int, endpoint=False)
            if self.cfg.move_sampling == "first":
                frames = np.linspace(
                    0, min(nf, total_time), total_time, dtype=int, endpoint=False
                )

            for t in frames:
                vert_data_copy = vert_data
                if t > 0:
                    vert_data_copy = vert_data + offset_data[t - 1]
                vert_datas.append(vert_data_copy)
                vert = vert_data_copy - np.mean(vert_data_copy, axis=0, keepdims=True)
                v_min = min(v_min, np.amin(vert))
                v_max = max(v_max, np.amax(vert))
            grids = []
            for vert_data in vert_datas:
                obj = trimesh.Trimesh(vert_data, face_data)
                obj = normalize(obj, v_min, v_max)
                voxel_grid: trimesh.voxel.VoxelGrid = vox_creation.voxelize(
                    obj, pitch=voxel_size
                )
                voxel_grid.fill()
                grid = voxel_grid.matrix
                padding_amounts = [
                    (floor((resolution - length) / 2), ceil((resolution - length) / 2))
                    for length in grid.shape
                ]
                grid = np.pad(grid, padding_amounts).astype(np.float32)
                grids.append(grid)
            grid = np.stack(grids)

        # Static objects
        else:
            mesh: trimesh.Trimesh = trimesh.load(path)
            coords = np.asarray(mesh.vertices)
            coords = coords - np.mean(coords, axis=0, keepdims=True)
            v_max = np.amax(coords)
            v_min = np.amin(coords)
            coords *= 0.95 / (max(abs(v_min), abs(v_max)))
            mesh.vertices = coords
            voxel_grid: trimesh.voxel.VoxelGrid = vox_creation.voxelize(
                mesh, pitch=voxel_size
            )
            voxel_grid.fill()
            grid = voxel_grid.matrix
            padding_amounts = [
                (floor((resolution - length) / 2), ceil((resolution - length) / 2))
                for length in grid.shape
            ]
            grid = np.pad(grid, padding_amounts).astype(np.float32)

        # Convert 0 regions to -1, so that the input is -1 or +1.
        grid[grid == 0] = -1

        grid = torch.tensor(grid).float()

        # Doing some sanity checks for 4D and 3D generations
        if self.cfg.mlp_config.params.move:
            assert (
                grid.shape[0] == total_time
                and grid.shape[1] == resolution
                and grid.shape[2] == resolution
                and grid.shape[3] == resolution
            )
            return grid, 0
        else:
            assert (
                grid.shape[0] == resolution
                and grid.shape[1] == resolution
                and grid.shape[2] == resolution
            )

        return grid[None, ...], 0

    def __len__(self):
        return len(self.mesh_files)


class WeightDataset(Dataset):
    def __init__(
        self, mlps_folder, wandb_logger, model_dims, mlp_kwargs, cfg, object_names=None
    ):
        """Load pre-trained MLPs that encode the shape of 3D objects"""

        self.mlps_folder = mlps_folder
        self.condition = cfg.transformer_config.params.condition
        files_list = list(os.listdir(mlps_folder))
        blacklist = {}
        if cfg.filter_bad:
            blacklist = set(np.genfromtxt(cfg.filter_bad_path, dtype=str))
        if object_names is None:
            self.mlp_files = [file for file in list(os.listdir(mlps_folder))]
        else:
            self.mlp_files = []
            for file in list(os.listdir(mlps_folder)):
                # Excluding black listed shapes
                if cfg.filter_bad and file.split("_")[1] in blacklist:
                    continue
                # Check if file is in corresponding split (train, test, val)
                # In fact, only train split is important here because we don't use test or val MLP weights
                if (
                    "_" in file
                    and (
                        file.split("_")[1] in object_names
                        or (file.split("_")[1] + "_" + file.split("_")[2])
                        in object_names
                    )
                ) or (file in object_names):
                    self.mlp_files.append(file)
        self.transform = None
        self.logger = wandb_logger
        self.model_dims = model_dims
        self.mlp_kwargs = mlp_kwargs
        if cfg.augment in ["permute", "permute_same", "sort_permute"]:
            self.example_mlp = get_mlp(mlp_kwargs)
        self.cfg = cfg
        if "first_weight_name" in cfg and cfg.first_weight_name is not None:
            self.first_weights = self.get_weights(
                torch.load(os.path.join(self.mlps_folder, cfg.first_weight_name))
            ).float()
        else:
            self.first_weights = torch.tensor([0])

    def get_weights(self, state_dict):
        weights = []
        shapes = []
        for weight in state_dict:
            shapes.append(np.prod(state_dict[weight].shape))
            weights.append(state_dict[weight].flatten().cpu())
        weights = torch.hstack(weights)
        prev_weights = weights.clone()

        # Some augmentation methods are available althougwe don't use them in the main paper
        if self.cfg.augment == "permute":
            weights = random_permute_flat(
                [weights], self.example_mlp, None, random_permute_mlp
            )[0]
        if self.cfg.augment == "sort_permute":
            example_mlp = generate_mlp_from_weights(weights, self.mlp_kwargs)
            weights = random_permute_flat(
                [weights], example_mlp, None, sorted_permute_mlp
            )[0]
        if self.cfg.augment == "permute_same":
            weights = random_permute_flat(
                [weights],
                self.example_mlp,
                int(np.random.random() * self.cfg.augment_amount),
                random_permute_mlp,
            )[0]
        if self.cfg.jitter_augment:
            weights += np.random.uniform(0, 1e-3, size=weights.shape)

        if self.transform:
            weights = self.transform(weights)
        # We also return prev_weights, in case you want to do permutation, we store prev_weights to sanity check later
        return weights, prev_weights

    def __getitem__(self, index):
        file = self.mlp_files[index]
        dir = join(self.mlps_folder, file)
        if os.path.isdir(dir):
            path1 = join(dir, "checkpoints", "model_final.pth")
            path2 = join(dir, "checkpoints", "model_current.pth")
            state_dict = torch.load(path1 if os.path.exists(path1) else path2)
        else:
            state_dict = torch.load(dir, map_location=torch.device("cpu"))

        weights, weights_prev = self.get_weights(state_dict)

        if self.cfg.augment == "inter":
            other_index = np.random.choice(len(self.mlp_files))
            other_dir = join(self.mlps_folder, self.mlp_files[other_index])
            other_state_dict = torch.load(other_dir)
            other_weights, _ = self.get_weights(other_state_dict)
            lerp_alpha = np.random.uniform(
                low=0, high=self.cfg.augment_amount
            )  # Prev: 0.3
            weights = torch.lerp(weights, other_weights, lerp_alpha)

        return weights.float(), weights_prev.float(), weights_prev.float()

    def __len__(self):
        return len(self.mlp_files)


class PartSemanticPointCloud(Dataset):
    def __init__(
        self,
        part_name: str,
        batch_size: int,
        coords: np.ndarray,
        labels: np.ndarray,
        occupancies: np.ndarray,
    ):
        super().__init__()
        self.part_name = part_name
        self.batch_size = batch_size
        self.coords = coords
        self.labels = labels
        self.occupancies = occupancies
        #print(f"Part: {part_name}")
        size = len(self.occupancies)
        occ_prob = self.occupancies.mean()
        #print(f"Number occupancies: {int(size*occ_prob)}")
        #print(f"Number not occupied Points: {int(size - size*occ_prob)}")

    def __len__(self):
        if self.coords.shape[0] == 0:
            return 0
        return max(1, self.coords.shape[0] // self.batch_size)

    def __getitem__(self, idx):
        """Randomly access a batch of points from the point cloud."""
        total_length = self.coords.shape[0]
        sample_size = self.batch_size

        random_indices = np.random.randint(total_length, size=sample_size)

        coords = self.coords[random_indices]
        labels = self.labels[random_indices]
        occs = self.occupancies[random_indices]

        output = {
            "coords": torch.from_numpy(coords).float(),
            "semantic_label": torch.from_numpy(labels).long(),
        }
        target = {"sdf": torch.from_numpy(occs).float()}

        return output, target


class SemanticPointCloud(Dataset):
    def __init__(
        self,
        on_surface_points: int,  # batch size
        pointcloud_path: str,
        pointcloud_expert_path: str,
        label_path: str,
        output_type: str = "occ",
        cfg=None,
    ):
        super().__init__()

        self.output_type = output_type
        self.cfg = cfg
        self.on_surface_points = on_surface_points

        pointcloud = None
        pointcloud_expert = None
        labels = None

        # Filled during initialization
        self.coords = None
        self.occupancies = None
        self.labels = None

        # assume mesh data
        if cfg.strategy == "save_pc":
            obj = self._load_mesh(pointcloud_path)
            coords, occupancies = self._preprocess_mesh_data(obj)
            pointcloud = np.hstack((coords, occupancies[:, None]))
            # save pointcloud as binary file
            self._save_pointcloud(pointcloud_path, coords, occupancies)
        # assume pre-computed pointcloud
        else:
            # load pointcloud that was saved as binary file
            pointcloud = self._load_binary_pointcloud(pointcloud_path)

        # Load point clouds with expert annotations
        pointcloud_expert = self._load_raw_pointcloud(pointcloud_expert_path)
        labels = self._load_raw_pointcloud_labels(label_path)

        # Compute normalization parameters from expert pointcloud itself
        mean, scale_factor = self._compute_normalization_params(pointcloud_expert)

        # Normalize pointcloud_expert to match pointcloud
        pointcloud_expert = self._normalize_pointcloud(
            pointcloud_expert, mean, scale_factor
        )

        # Rotate 90 degrees around Y axis to match generated pointcloud
        # x_new = z_old, z_new = -x_old
        x = pointcloud_expert[:, 0].copy()
        pointcloud_expert[:, 0] = pointcloud_expert[:, 2]
        pointcloud_expert[:, 2] = -x

        # print(
        #     "Pointcloud Expert:",
        #     pointcloud_expert.min(axis=0),
        #     pointcloud_expert.max(axis=0),
        #     mean,
        #     scale_factor,
        # )

        pointcloud_coords = pointcloud[:, :3]
        mean, scale_factor = self._compute_normalization_params(pointcloud_coords)
        pointcloud[:, :3] = self._normalize_pointcloud(
            pointcloud_coords, mean, scale_factor
        )

        # print(
        #     "Pointcloud:",
        #     pointcloud_coords.min(axis=0),
        #     pointcloud_coords.max(axis=0),
        #     mean,
        #     scale_factor,
        # )

        # Most likely there will be a mismatch in the number of points
        # apply nearest neighbor matching, to align pointcloud and labels
        if pointcloud.shape != labels.shape:
            labels = self._nearest_neighbor_matching(
                pointcloud, pointcloud_expert, labels
            )

        if cfg and cfg.get("shape_modify") == "half":
            self._apply_half_shape_filter()

        self.coords = pointcloud[:, :3]
        self.occupancies = pointcloud[:, 3]
        self.labels = labels * self.occupancies

    def get_part_specific_pointcloud_datasets(
        self,
    ):
        part_pointclouds = {}
        for idx, part_name in enumerate(self.cfg.label_names):
            # Zero out rest of the object
            part_occupancies = (self.labels == idx + 1) * self.occupancies
            part_pointclouds.update({
                part_name: PartSemanticPointCloud(
                    part_name,
                    self.on_surface_points,
                    self.coords,
                    self.labels,
                    part_occupancies,
                )})
        return part_pointclouds
        

    def _nearest_neighbor_matching(self, pointcloud, pointcloud_expert, labels):
        nn_matcher = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
        nn_matcher.fit(pointcloud_expert)

        pointcloud_coords = pointcloud[:, :3]
        # find nearest neighbor for each point in pointcloud
        distances, indices = nn_matcher.kneighbors(pointcloud_coords, return_distance=True)
        print(f" alignment error ({np.max(distances)}).")
        closest_expert_indices = indices.flatten()
        matched_labels = labels[closest_expert_indices]
        return matched_labels

    def _compute_normalization_params(self, pointcloud):
        mean = np.mean(pointcloud, axis=0, keepdims=True)
        pointcloud_centered = pointcloud - mean
        v_max, v_min = np.amax(pointcloud_centered), np.amin(pointcloud_centered)
        scale_factor = 0.5 * 0.95 / (max(abs(v_min), abs(v_max)))
        return mean, scale_factor

    def _normalize_pointcloud(self, pointcloud, mean, scale_factor):
        pointcloud -= mean
        pointcloud *= scale_factor
        return pointcloud

    def _load_mesh(self, path: str) -> trimesh.Trimesh:
        """Loads a mesh file and applies normalization."""
        obj: trimesh.Trimesh = trimesh.load(path)

        # Normalization logic
        vertices = obj.vertices
        mean, scale_factor = self._compute_normalization_params(vertices)
        vertices = self._normalize_pointcloud(vertices, mean, scale_factor)
        obj.vertices = vertices

        return obj

    def _preprocess_mesh_data(self, obj: trimesh.Trimesh):
        """Samples points from the normalized mesh and calculates occupancy labels."""
        total_points = self.cfg.n_points

        # Point Sampling
        points_uniform = np.random.uniform(-0.5, 0.5, size=(total_points, 3))
        points_surface = obj.sample(total_points)
        points_surface += 0.01 * np.random.randn(total_points, 3)
        all_points = np.concatenate([points_surface, points_uniform], axis=0)

        # Calculate Labels (Occupancies)
        coords, occupancies = self._calculate_occupancies(all_points, obj)
        return coords, occupancies

    def _calculate_occupancies(
        self, points: np.ndarray, obj: trimesh.Trimesh
    ):
        import igl

        inside_surface_values = igl.fast_winding_number_for_meshes(
            obj.vertices, obj.faces, points
        )

        thresh = 0.5
        occupancies = np.piecewise(
            inside_surface_values,
            [inside_surface_values < thresh, inside_surface_values >= thresh],
            [0, 1],
        )
        return points, occupancies

    def _save_pointcloud(self, path: str, coords: np.ndarray, occupancies: np.ndarray):
        pc_folder = SemanticPointCloud.get_pc_folder_name(self.cfg)
        os.makedirs(pc_folder, exist_ok=True)

        point_cloud_data = np.hstack((coords, occupancies[:, None]))

        save_path = os.path.join(pc_folder, os.path.basename(path) + ".npy")
        np.save(save_path, point_cloud_data)

    def _load_binary_pointcloud(self, path: str):
        """Loads a binary point cloud file (.npy)."""
        pc_folder = SemanticPointCloud.get_pc_folder_name(self.cfg)
        load_path = os.path.join(pc_folder, os.path.basename(path) + ".npy")
        point_cloud = np.load(load_path)

        mean, scale_factor = self._compute_normalization_params(point_cloud[:, :3])
        point_cloud[:, :3] = self._normalize_pointcloud(
            point_cloud[:, :3], mean, scale_factor
        )
        return point_cloud

    def _load_raw_pointcloud_labels(self, path: str):
        return np.genfromtxt(path)

    def _load_raw_pointcloud(self, path: str):
        if path.endswith(".npy"):
            point_cloud = np.load(path)
        else:
            point_cloud = np.genfromtxt(path)
        return point_cloud

    def _apply_half_shape_filter(self):
        included_points = self.pointcloud[:, 0] < 0
        self.pointcloud = self.pointcloud[included_points]

        included_points = self.pointcloud_expert[:, 0] < 0
        self.pointcloud_expert = self.pointcloud_expert[included_points]
        self.labels = self.labels[included_points]

    @staticmethod
    def get_pc_folder_name(cfg: DictConfig) -> str:
        base_dir = os.path.dirname(cfg.dataset_folder)
        # base_dir = cfg.dataset_folder

        n_points = str(cfg.n_points)
        folder_name = f"{base_dir}_{n_points}_pc"

        return folder_name

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        """Randomly access a batch of points from the point cloud."""
        total_length = self.coords.shape[0]
        sample_size = self.on_surface_points

        random_indices = np.random.randint(total_length, size=sample_size)

        coords = self.coords[random_indices]
        labels = self.labels[random_indices]
        occs = self.occupancies[random_indices]

        output = {
            "coords": torch.from_numpy(coords).float(),
            #"semantic_label": torch.from_numpy(labels).long()
        }
        #TODO: Compare sdf and labels occupancy values
        # Then remove sdf from target and semantic_label output
        target = {#"sdf": torch.from_numpy(occs).float(),
                  "semantic_label": torch.from_numpy(labels).long()
        }
        return output, target


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)

    meta_data_path = "data/shapenetpart/PartAnnotation/metadata.json"
    meta_data = load_meta_data(meta_data_path)
    CATEGORY = "Airplane"  # Airplane, Chair, Car
    directory = meta_data[CATEGORY]["directory"]
    label_names = meta_data[CATEGORY]["lables"]
    print("Label names:", label_names)

    file_id = "1a32f10b20170883663e90eaf6b4ca52"
    # precomputed hyperdiffusion pointcloud based on meshes
    pointcloud_path = f"data/baseline/{directory}/"

    # shapenet part
    pointcloud_expert_path = f"data/shapenetpart/PartAnnotation/{directory}/points/"
    pointcloud_expert_label_path = (
        f"data/shapenetpart/PartAnnotation/{directory}/expert_verified/points_label/"
    )

    dataset_semantic_pc = SemanticPointCloud(
        on_surface_points=2048,
        pointcloud_path=pointcloud_path + f"{file_id}.obj",
        pointcloud_expert_path=pointcloud_expert_path + f"{file_id}.pts",
        label_path=pointcloud_expert_label_path + f"{file_id}.seg",
        output_type="occ",
        cfg=DictConfig(
            {
                "n_points": 2048,
                "strategy": "first_weights",
                "output_type": "occ",
                "dataset_folder": pointcloud_path,
                "label_names": label_names,
            }
        ),
    )

    item = dataset_semantic_pc.__getitem__(0)[0]
    coords = item["coords"].numpy()
    labels = item["semantic_label"].numpy()
    print(item)
    # map numeric values to labels
    labels = np.array([label_names[i - 1] for i in labels])
    visualize_pointcloud_3d(coords, labels)

    # Visualize the pointclouds for each part
    part_specific_pointcloud_datasets = (
        dataset_semantic_pc.get_part_specific_pointcloud_datasets()
    )
    for part_name, dataset in part_specific_pointcloud_datasets.items():
        print(part_name)
        item = dataset.__getitem__(0)[0]
        coords = item["coords"].numpy()
        labels = item["semantic_label"].numpy()
        labels = np.array([label_names[i - 1] for i in labels])
        visualize_pointcloud_3d(coords, labels)
