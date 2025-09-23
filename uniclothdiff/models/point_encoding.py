

import torch
import torch.nn as nn

from uniclothdiff.models.common import KNNGrouper, PatchEncoder


class PointcloudEmbed(nn.Module):
    def __init__(
        self,
        out_channels=1024,
        num_groups=256,
        group_size=64,
        radius: float = None,
        centralize_features=False,
    ):
        super().__init__()
        # self.in_channels = in_channels
        self.out_channels = out_channels

        self.grouper = KNNGrouper(
            num_groups,
            group_size,
            radius=radius,
            centralize_features=centralize_features,
        )

        self.patch_encoder = PatchEncoder(3, out_channels, [128, 512])
        
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.out_channels)
        )  
        

    def forward(self, coords: torch.Tensor, features: torch.Tensor):
        # 1. group points
        patches = self.grouper(coords, features)
        # 2. patch encoder
        patch_features = patches["features"][:, :, :, :3]  # [B, L, K, C_in]
        x = self.patch_encoder(patch_features)
        # 3. add positional embeddings
        centers = patches["centers"]
        pos_embed = self.pos_embed(centers)
        x = x + pos_embed
        
        return x
