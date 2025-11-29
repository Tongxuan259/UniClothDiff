import os
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset

import h5py
import open3d as o3d

from uniclothdiff.registry import DATASETS

@DATASETS.register_module()
class ClothStateEstDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        mode='train',
        num_sample_points=10000,
        template_mesh_path=None,
        camera_params=None,
        do_camera_pose_augmentation=True,
        do_point_cloud_augmentation=True,
        r_range=1.5,
        t_range=0.0025,
        voxel_downsample_size=0.005, 
        points_jitter_sigma=0.0005,
        points_drop_ratio=0.0,
    ):
        self.mode = mode
        self.data_dir = data_dir
        
        self.data_files = sorted(os.listdir(self.data_dir))
        num_data_files = len(self.data_files)
        
        self.num_sample_points = num_sample_points
        self.camera_params = camera_params
        self.do_camera_pose_augmentation = do_camera_pose_augmentation
        self.do_point_cloud_augmentation = do_point_cloud_augmentation
        self.r_range = r_range
        self.t_range = t_range
        self.voxel_downsample_size = voxel_downsample_size
        self.points_jitter_sigma = points_jitter_sigma
        self.points_drop_ratio = points_drop_ratio

        self._load_template(template_mesh_path)
        
        if mode == 'train':
            self.data_files = self.data_files[:int(num_data_files * 0.95)]                
        else:
            self.data_files = self.data_files[int(num_data_files * 0.95):]
        self.num_samples = len(self.data_files)

    def __len__(self):
        return self.num_samples
    
    def _load_template(self, template_path):
        """load template mesh"""
        with open(template_path, 'rb') as file:
            template_data = pickle.load(file)
            self.q_template = torch.tensor(
                template_data['points'],
                dtype=torch.float32
            )
    
    def sample_points(self, pcd):
        """resample points to self.num_sample_points"""
        num_points = pcd.shape[0]
        
        if num_points == self.num_sample_points:
            return pcd
        elif num_points > self.num_sample_points:
            indices = torch.randperm(num_points)[:self.num_sample_points]
            return pcd[indices]
        else:
            indices = torch.randint(num_points, (self.num_sample_points,))
            return pcd[indices]    
    
    def depth_to_point_cloud(self, depth):
        """convert depth map to point cloud
        
        Args:
            depth: shape (H, W), depth map
            
        Returns:
            points: shape (N, 3), N points
        """
        height, width = depth.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # convert to normalized plane
        x_norm = (x - self.camera_params['cx']) / self.camera_params['fx']
        y_norm = (y - self.camera_params['cy']) / self.camera_params['fy']
        
        # calculate 3D points
        z = depth
        x = x_norm * z
        y = y_norm * z
        
        # organize valid points to point cloud
        mask = z > 0
        points = np.stack([x[mask], y[mask], z[mask]], axis=1)
        
        return points
    
    def camera_to_world(self, points_camera, c2w_matrix):
        """convert points in camera coordinate to world coordinate"""
        return points_camera @ c2w_matrix.T
    
    def to_open3d_pcd(self, points):
        """convert points to o3d point cloud"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    
    def augment_camera_pose(
        self,
        c2w_matrix: np.ndarray,
        rot_range: float,    # rotation range (degree)
        trans_range: float,  # translation range (meter)
    ) -> np.ndarray:
        """Augment camera pose.
        
        Args:
            c2w_matrix: shape (4, 4), camera-to-world transformation matrix
            rot_range: maximum rotation angle (degree)
            trans_range: maximum translation distance (meter)
        
        Returns:
            augmented_matrix: shape (4, 4), augmented transformation matrix
        """
        # decompose c2w_matrix to rotation and translation
        R = c2w_matrix[:3, :3]
        t = c2w_matrix[:3, 3]
        
        # random rotation perturbation (Euler angles, radians)
        rot_noise = np.random.uniform(
            -np.deg2rad(rot_range),
            np.deg2rad(rot_range),
            size=3
        )
        
        # convert Euler angles to rotation matrix
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rot_noise[0]), -np.sin(rot_noise[0])],
            [0, np.sin(rot_noise[0]), np.cos(rot_noise[0])]
        ])
        
        Ry = np.array([
            [np.cos(rot_noise[1]), 0, np.sin(rot_noise[1])],
            [0, 1, 0],
            [-np.sin(rot_noise[1]), 0, np.cos(rot_noise[1])]
        ])
        
        Rz = np.array([
            [np.cos(rot_noise[2]), -np.sin(rot_noise[2]), 0],
            [np.sin(rot_noise[2]), np.cos(rot_noise[2]), 0],
            [0, 0, 1]
        ])
        
        R_noise = Rz @ Ry @ Rx
        
        # random translation perturbation
        t_noise = np.random.uniform(-trans_range, trans_range, size=3)
        
        # apply perturbation
        R_aug = R @ R_noise
        t_aug = t + t_noise
        
        # build augmented transformation matrix
        augmented_matrix = np.eye(4)
        augmented_matrix[:3, :3] = R_aug
        augmented_matrix[:3, 3] = t_aug
        
        return augmented_matrix

    def augment_multi_view_poses(
        self,
        c2w_matrices: np.ndarray,
    ) -> np.ndarray:
        """Augment multi-view camera poses.
        
        Args:
            c2w_matrices: shape (N, 4, 4), N camera poses
            consistent: whether to use the same augmentation parameters for all views
            
        Returns:
            augmented_matrices: shape (N, 4, 4), augmented transformation matrices
        """      
        num_views = len(c2w_matrices)
        augmented_matrices = np.zeros_like(c2w_matrices)
        
        for i in range(num_views):
            augmented_matrices[i] = self.augment_camera_pose(
                c2w_matrices[i],
                rot_range=self.r_range,
                trans_range=self.t_range
            )
    
        return augmented_matrices
    
    
    def augment_point_cloud(self, points):
        """augment point cloud
        
        Args:
            points: shape (N, 3), N points
            
        Returns:
            augmented_points: shape (N, 3), augmented points
        """
        
        points = points.copy()
    
        # 1. Add Gaussian noise (point-level perturbation)
        if self.points_jitter_sigma > 0:
            noise = np.random.normal(0, self.points_jitter_sigma, points.shape)
            points = points + noise
        
        # 2. Random drop points
        if self.points_drop_ratio > 0:
            num_points = len(points)
            num_keep = int(num_points * (1 - self.points_drop_ratio))
            indices = np.random.choice(num_points, num_keep, replace=False)
            points = points[indices]
        
        return points

    def _getitem_processed_data(self, idx):
        file_name = self.data_files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        # file_path = self.data_files[idx]
        try:
            with h5py.File(file_path, 'r') as raw_data:
                points = raw_data['points'][:]
                q = raw_data['q'][:]
                
            if self.do_point_cloud_augmentation:
                points = self.augment_point_cloud(points)
                
            points = self.sample_points(points)
            
            return {
                'q_gt': torch.tensor(q, dtype=torch.float32),
                'pcd': torch.tensor(points, dtype=torch.float32),
                'q_temp': self.q_template
            }
                
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        return self._getitem_processed_data(idx)
    
        