import os
import pickle
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import h5py
import open3d as o3d
from uniclothdiff.registry import DATASETS


@DATASETS.register_module()
class ClothStateEstDataset(Dataset):
    def __init__(
            self, 
            data_dir, 
            resolution=(50, 50),
            min_q=[-1.0, -1.0, -1.0],
            max_q=[ 1.0,  1.0,  1.0],
            num_sample_points=5000,
            mode='train',
            reshape_2d=False,
            template_mesh_path="misc/cloth_simple.pkl",
            pcd_denoise=True,
        ):
        self.mode = mode
        self.data_dir = data_dir
        self.resolution = resolution
        self.data_files = sorted(os.listdir(self.data_dir))
        self.reshape_2d = reshape_2d
        self.num_sample_points = num_sample_points
        self.pcd_denoise = pcd_denoise
       
        
       
        self.num_samples = len(self.data_files)

        if mode == 'valid':
            self.data_files = self.data_files[self.num_samples * 0.95:]

        
        self.min_q = torch.tensor(np.array(min_q), dtype=torch.float32)
        self.max_q = torch.tensor(np.array(max_q), dtype=torch.float32)
        
        with open(template_mesh_path, 'rb') as file:
            template_data = pickle.load(file)
            q_template = torch.tensor(
                template_data['points'],
                dtype=torch.float32
            )
            if self.reshape_2d:
                q_template = q_template.reshape(self.resolution[0], self.resolution[1], -1).permute(2, 0, 1)
            
            self.q_temp= q_template

    def __len__(self):
        return self.num_samples
    
    
    def pcd_transform(self, pcd, nb_neighbors=250, std_ratio=0.1):
        
        raw_pcd = o3d.geometry.PointCloud()
        raw_pcd.points = o3d.utility.Vector3dVector(pcd.numpy())
        raw_pcd.voxel_down_sample(voxel_size=0.005)
        pcd = torch.from_numpy(np.asarray(raw_pcd.points)).to(torch.float32)
            
        
        # pcd centeralize
        center = torch.mean(pcd, dim=0)
        centered_pcd = pcd - center
        
        # pcd resample
        num_points = centered_pcd.shape[0]
    
        if num_points == self.num_sample_points:
            return centered_pcd
        elif num_points > self.num_sample_points:
            indices = torch.randperm(num_points)[:self.num_sample_points]
            return centered_pcd[indices]
        else:
            indices = torch.randint(num_points, (self.num_sample_points,))
            return centered_pcd[indices]
        
    
    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        
        file_path = os.path.join(self.data_dir, file_name)
        
        try:
            with h5py.File(file_path, 'r') as raw_data:
                pcd = raw_data['points'][:]
                q = raw_data['q'][:]
                
                q = torch.tensor(q, dtype=torch.float32)
                pcd = torch.tensor(pcd, dtype=torch.float32)
                
                if self.reshape_2d:
                    q = q.reshape(self.resolution[0], self.resolution[1], -1).permute(2, 0, 1)
                
                pcd = self.pcd_transform(pcd)
                
                return {
                   "pcd": pcd,
                   "q_gt": q,
                   'q_temp':self.q_temp
                }
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None