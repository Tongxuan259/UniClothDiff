import argparse
import pickle
import numpy as np
import trimesh
import torch
import torch.nn as nn
import open3d as o3d
from scipy.spatial import cKDTree

from torkit3d.nn.functional import batch_index_select
from torkit3d.ops.sample_farthest_points import sample_farthest_points


# ----------------------------------------------------------------------
# KNN grouping utilities
# ----------------------------------------------------------------------
def knn_points(
    query: torch.Tensor,
    key: torch.Tensor,
    k: int,
    sorted: bool = False,
    transpose: bool = False,
):
    """
    Compute k-nearest neighbors between `query` and `key`.

    Args:
        query: (B, N1, C) tensor of query points.
        key:   (B, N2, C) tensor of key points.
        k:     number of neighbors.
        sorted: whether to return neighbors sorted by distance.
        transpose: if True, treat input as (B, C, N) and transpose.

    Returns:
        knn_dist: (B, N1, k) distances to neighbors.
        knn_ind:  (B, N1, k) indices of neighbors in `key`.
    """
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)

    # Pairwise distances: (B, N1, N2)
    distance = torch.cdist(query, key)

    if k == 1:
        knn_dist, knn_ind = torch.min(distance, dim=2, keepdim=True)
    else:
        knn_dist, knn_ind = torch.topk(distance, k, dim=2, largest=False, sorted=sorted)

    return knn_dist, knn_ind


class KNNGrouper(nn.Module):
    """
    Group a point cloud into local patches using FPS centers + KNN neighbors.

    1) Use farthest point sampling (FPS) to select `num_groups` centers.
    2) For each center, find `group_size` nearest neighbors.
    """

    def __init__(self, num_groups: int, group_size: int, radius=None, centralize_features: bool = False):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.radius = radius
        self.centralize_features = centralize_features

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, use_fps: bool = True):
        """
        Args:
            xyz: (B, N, 3) point cloud.
            features: unused, kept for possible extension.
            use_fps: whether to use FPS to choose centers.

        Returns:
            knn_idx_flat: (B * num_groups * group_size,) flattened neighbor indices.
            nbr_xyz:      (B, num_groups, group_size, 3) grouped neighbor coordinates.
        """
        batch_size, num_points, _ = xyz.shape

        with torch.no_grad():
            if use_fps:
                # FPS center indices: (B, G)
                fps_idx = sample_farthest_points(xyz.float(), self.num_groups)
                # Center coordinates: (B, G, 3)
                centers = batch_index_select(xyz, fps_idx, dim=1)
            else:
                # Use the first `num_groups` points as centers
                fps_idx = torch.arange(self.num_groups, device=xyz.device)
                fps_idx = fps_idx.expand(batch_size, -1)
                centers = xyz[:, : self.num_groups]

            # KNN indices for each center: (B, G, K)
            _, knn_idx = knn_points(centers, xyz, self.group_size, sorted=True)

        # Convert (B, G, K) indices into flat indices for (B * N) points
        batch_offset = torch.arange(batch_size, device=xyz.device) * num_points
        batch_offset = batch_offset.view(-1, 1, 1)  # (B, 1, 1)
        knn_idx_flat = (knn_idx + batch_offset).reshape(-1)  # (B * G * K,)

        # Gather neighbor coordinates and reshape back to (B, G, K, 3)
        nbr_xyz = xyz.reshape(-1, 3)[knn_idx_flat]
        nbr_xyz = nbr_xyz.reshape(batch_size, self.num_groups, self.group_size, 3)

        return knn_idx_flat, nbr_xyz


# ----------------------------------------------------------------------
# Voronoi-style partition: FPS centers + nearest-center assignment
# ----------------------------------------------------------------------
def split_point_cloud_with_voronoi(point_cloud: torch.Tensor, n_patches: int = 100):
    """
    Partition a point cloud into Voronoi-style patches:
    - Use FPS to sample `n_patches` centers.
    - Assign each point to its nearest center (Euclidean distance).

    Args:
        point_cloud: (1, N, 3) torch tensor.
        n_patches:   number of patches / centers.

    Returns:
        patches:     list[np.ndarray], each of shape (Pi, 3), patch coordinates.
        patches_idx: list[np.ndarray], each of shape (Pi,), indices in original point cloud.
        centers:     (1, n_patches, 3) torch tensor of center coordinates.
        fps_idx:     (1, n_patches) torch tensor of center indices.
    """
    # FPS centers
    fps_idx = sample_farthest_points(point_cloud.float(), n_patches)  # (1, n_patches)
    centers = batch_index_select(point_cloud, fps_idx, dim=1)         # (1, n_patches, 3)

    # KD-tree on centers (drop batch dim)
    centers_np = centers[0].cpu().numpy()
    tree = cKDTree(centers_np)

    patches = [[] for _ in range(n_patches)]
    patches_idx = [[] for _ in range(n_patches)]

    # Assign each point to nearest center
    points_np = point_cloud[0].cpu().numpy()
    for point_idx, point in enumerate(points_np):
        _, nearest_center_idx = tree.query(point)
        patches[nearest_center_idx].append(point)
        patches_idx[nearest_center_idx].append(point_idx)

    # Convert lists to numpy arrays and drop empty patches (if any)
    patches = [np.array(patch, dtype=np.float32) for patch in patches if len(patch) > 0]
    patches_idx = [np.array(patch, dtype=np.int64) for patch in patches_idx if len(patch) > 0]

    return patches, patches_idx, centers, fps_idx


# ----------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------
def visualize_voronoi_patches(points: np.ndarray, patches_idx: list):
    """
    Visualize Voronoi-style patches with random colors.

    Args:
        points:      (N, 3) original point cloud.
        patches_idx: list[np.ndarray], each array holds indices of a patch.
    """
    num_points = points.shape[0]
    colors = np.zeros((num_points, 3), dtype=np.float32)

    # Assign a random color to each patch
    for patch_indices in patches_idx:
        color = np.random.rand(1, 3).astype(np.float32)
        colors[patch_indices] = color

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([pcd, coord])


def visualize_knn_patches(points: np.ndarray, point_patch_idx: np.ndarray):
    """
    Visualize KNN patches (fixed-size neighborhoods) with random colors.

    Args:
        points:          (N, 3) original point cloud.
        point_patch_idx: (num_patches, patch_size) indices into `points`.
    """
    num_points = points.shape[0]
    num_patches, patch_size = point_patch_idx.shape

    colors = np.zeros((num_points, 3), dtype=np.float32)

    # Assign a random color to each KNN patch
    for i in range(num_patches):
        idx = point_patch_idx[i]
        color = np.random.rand(1, 3).astype(np.float32)
        colors[idx] = color

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([pcd, coord])


# ----------------------------------------------------------------------
# Main: cloth → patches (Voronoi or KNN)
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Patchify cloth mesh into Voronoi or KNN patches."
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        required=True,
        help="Path to cloth mesh file (e.g., sweater.obj).",
    )
    parser.add_argument(
        "--patchify_method",
        type=str,
        default="voronoi",
        choices=["voronoi", "knn"],
        help="Patchification method: 'voronoi' or 'knn'.",
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        default=100,
        help="Number of patches / FPS centers.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=25,
        help="KNN patch size (only used when patchify_method='knn').",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="cloth",
        help="Prefix for output files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for torch operations.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Visualize the resulting patches.",
    )

    args = parser.parse_args()

    # Select device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load mesh and build point cloud
    # ------------------------------------------------------------------
    print(f"[1/3] Loading mesh from: {args.mesh_path}")
    mesh = trimesh.load(args.mesh_path)
    points = mesh.vertices.astype(np.float32)  # (N, 3)

    # Simple normalization: scale and center at origin
    points *= 0.5
    center = np.mean(points, axis=0, keepdims=True)
    points -= center

    print(f"Point cloud shape: {points.shape}")  # (N, 3)

    # Convert to torch and add batch dimension
    points_tensor = torch.from_numpy(points[None]).to(device)  # (1, N, 3)

    method = args.patchify_method.lower()
    print(f"[2/3] Patchify method: {method}")

    # ------------------------------------------------------------------
    # 2. Patchify (Voronoi or KNN)
    # ------------------------------------------------------------------
    if method == "voronoi":
        # Voronoi-style: no patch_size needed
        print(f"  - Using Voronoi-style patches with {args.num_patches} centers (variable patch sizes).")
        patched_points, patched_idx, centers, fps_idx = split_point_cloud_with_voronoi(
            points_tensor, n_patches=args.num_patches
        )

        out_path = f"assets/{args.output_prefix}_voronoi_template.pkl"
        data = {
            "patch_points": patched_points,            # list[np.ndarray], each (Pi, 3)
            "patch_index": patched_idx,                # list[np.ndarray], each (Pi,)
            "centers": centers[0].cpu().numpy(),       # (num_patches, 3)
            "center_idx": fps_idx[0].cpu().numpy(),    # (num_patches,)
            "points": points,                          # (N, 3) normalized cloth points
        }
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[3/3] Saved Voronoi patches to: {out_path}")

        if args.vis:
            print("[Vis] Visualizing Voronoi patches...")
            visualize_voronoi_patches(points, patched_idx)

    elif method == "knn":
        # KNN-style: fixed patch_size needed
        print(f"  - Using KNN patches with {args.num_patches} groups, patch_size={args.patch_size}.")
        grouper = KNNGrouper(
            num_groups=args.num_patches,
            group_size=args.patch_size,
            radius=None,
            centralize_features=False,
        ).to(device)

        # Dummy features: not used
        dummy_features = torch.ones(1, points_tensor.shape[1], 1, device=device)

        point_idx_flat, patch_xyz = grouper(points_tensor, dummy_features)
        point_idx = point_idx_flat.reshape(1, args.num_patches, args.patch_size)  # (1, G, K)
        point_patch_idx = point_idx[0].cpu().numpy()  # (G, K)

        out_path = f"assets/{args.output_prefix}_knn_patch_idx.npy"
        np.save(out_path, point_patch_idx)
        print(f"[3/3] Saved KNN patch indices to: {out_path}")

        if args.vis:
            print("[Vis] Visualizing KNN patches...")
            visualize_knn_patches(points, point_patch_idx)

    print("[Done] Patchification finished.")


if __name__ == "__main__":
    main()
