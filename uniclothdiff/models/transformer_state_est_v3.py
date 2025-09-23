# Copyright 2024 the Latte Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional
import numpy as np
import pickle
import torch
from torch import nn

import numpy as np

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from uniclothdiff.models.positional_encoding import PachifiedEmbed
from uniclothdiff.models.point_encoding import PointcloudEmbed
from uniclothdiff.registry import MODELS

@MODELS.register_module()
class TransformerStateEstV3Model(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    @register_to_config
    def __init__(
        self,
        pc_model: str ="eva02_large_patch14_448",
        pc_enc_checkpoint: Optional[str] = None,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = 1024,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = 1000,
        norm_type: str = "ada_norm_zero",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        pcd_embed_dim: int = 1024,
        num_in_frames: int = 2,
        num_out_frames: int = 1,
        patchified_input: bool = False,
        num_groups: int = 256,
        group_size: int = 64,
        patch_file: str = None,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.patchified_input = patchified_input
        self.num_in_frames = num_in_frames
        self.num_out_frames = num_out_frames

        # 1. prepare patch index and patch offset for template
        with open(patch_file, 'rb') as file:
            data = pickle.load(file)
        patch_index = data['patch_index']
        coords = torch.tensor(data['points'][None], dtype=torch.float32)
        self.patch_offset = torch.cat([
            torch.tensor([0], dtype=torch.int16),
            torch.cumsum(
                torch.tensor([patch.shape[0] for patch in patch_index]), dim=0
            )
        ])
        self.patch_size = torch.tensor([
            self.patch_offset[i] - self.patch_offset[i-1] \
                for i in range(1, self.patch_offset.shape[0])
        ])
        self.patch_index = torch.cat([
            torch.tensor(patch, dtype=torch.int16) for patch in patch_index
        ])
        self.inv_patch_index = torch.argsort(self.patch_index)

        centers = torch.tensor(data['centers'], dtype=torch.float32)[None]
        
        self.pos_embed = PachifiedEmbed(
            in_channels=in_channels,
            embed_dim=inner_dim,
            patch_index=self.patch_index,
            patch_offset=self.patch_offset,
            patch_centers=centers
        )
        self.interp_index, self.interp_weight = self.compute_interp_weights(query=coords, key=centers)

        
        # 1.5. Define point cloud encoder
        self.pc_embed = PointcloudEmbed(
            out_channels=pcd_embed_dim,
            num_groups=num_groups,
            group_size=group_size,
        )
        
        
        # 2. Define spatial transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for d in range(num_layers)
            ]
        )

        
        # 3. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        # self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)

        self.proj_out = nn.Sequential(
            nn.Linear(inner_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.out_channels),
        )

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def knn_points(self,
        query: torch.Tensor,
        key: torch.Tensor,
        k: int,
        sorted: bool = False,
        transpose: bool = False,
    ):
        if transpose:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
        # Compute pairwise distances, [B, N1, N2]
        distance = torch.cdist(query, key)
        if k == 1:
            knn_dist, knn_ind = torch.min(distance, dim=2, keepdim=True)
        else:
            knn_dist, knn_ind = torch.topk(distance, k, dim=2, largest=False, sorted=sorted)
        return knn_dist, knn_ind
        
    
    def compute_interp_weights(
        self,
        query: torch.Tensor, 
        key: torch.Tensor, 
        k=3, 
        eps=1e-8
    ):
        dist, idx = self.knn_points(query, key, k)
        inv_dist = 1.0 / torch.clamp(dist.square(), min=eps)
        normalizer = torch.sum(inv_dist, dim=2, keepdim=True)
        weight = inv_dist / normalizer  # [B, Nq, K]
        return idx, weight


    def interpolate_features(
        self,
        x: torch.Tensor, 
        index: torch.Tensor, 
        weight: torch.Tensor
    ):
        """
        Interpolates features based on the given index and weight.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_keys, num_features).
            index (torch.Tensor): The index tensor of shape (batch_size, num_queries, K).
            weight (torch.Tensor): The weight tensor of shape (batch_size, num_queries, K).

        Returns:
            torch.Tensor: The interpolated features tensor of shape (batch_size, num_queries, num_features).
        """
        B, Nq, K = index.shape
        batch_offset = torch.arange(B, device=x.device).reshape(-1, 1, 1) * x.shape[1]
        index_flat = (index + batch_offset).flatten()  # [B*Nq*K]
        _x = x.flatten(0, 1)[index_flat].reshape(B, Nq, K, x.shape[-1])
        return (_x * weight.unsqueeze(-1)).sum(-2)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        enable_temporal_attentions: bool = True,
        return_dict: bool = True,
    ):  
        batch_size = hidden_states.shape[0]

        hidden_states, input_hidden_states = self.pos_embed(hidden_states)  # alrady add positional embeddings
        
        # Prepare point cloud embeddings for spatial block
        encoder_hidden_states = self.pc_embed(encoder_hidden_states, encoder_hidden_states)
        # for inference only
        if len(timestep.shape) < 1:
            timestep = timestep.expand(batch_size)
        # Prepare timesteps for spatial block
        timestep_spatial = timestep

        # Spatial transformer blocks
        for i, spatial_block in enumerate(self.transformer_blocks):
            # spatial block
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    spatial_block,
                    hidden_states,
                    None,  # attention_mask
                    encoder_hidden_states,   # encoder_hidden_states, remember to modify
                    encoder_attention_mask,
                    timestep_spatial,
                    None,  # cross_attention_kwargs
                    None,  # class_labels
                    use_reentrant=False,
                )
            else:
                hidden_states = spatial_block(
                    hidden_states,
                    None,  # attention_mask
                    encoder_hidden_states, 
                    encoder_attention_mask,
                    timestep_spatial,
                    None,  # cross_attention_kwargs
                    None,  # class_labels
                )
            
        hidden_states = self.norm_out(hidden_states)
        
        hidden_states = self.interpolate_features(
            hidden_states, 
            self.interp_index.repeat(hidden_states.size(0), 1, 1).cuda(), 
            self.interp_weight.repeat(hidden_states.size(0), 1, 1).cuda(),
        )
            
        hidden_states = hidden_states + input_hidden_states[:, self.inv_patch_index, :]
            
        hidden_states = self.proj_out(hidden_states)

        ## unpatchify
        output = hidden_states

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)