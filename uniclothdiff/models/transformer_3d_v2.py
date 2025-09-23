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
from diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import PatchEmbed
# from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle



from uniclothdiff.models.positional_encoding import ActionEmbedding, PachifiedEmbed
from uniclothdiff.registry import MODELS

# @MODELS.register_module()
class Transformer3Dv2Model(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    """
    A 3D Transformer model for video-like data, paper: https://arxiv.org/abs/2401.03048, offical code:
    https://github.com/Vchitect/Latte

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input.
        out_channels (`int`, *optional*):
            The number of channels in the output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        patch_size (`int`, *optional*):
            The size of the patches to use in the patch embedding layer.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states. During inference, you can denoise for up to but not more steps than
            `num_embeds_ada_norm`.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The type of normalization to use. Options are `"layer_norm"` or `"ada_layer_norm"`.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon value to use in normalization layers.
        caption_channels (`int`, *optional*):
            The number of channels in the caption embeddings.
        video_length (`int`, *optional*):
            The number of frames in the video-like data.
    """
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = 1024,
        attention_bias: bool = False,
        sample_size: int = 64,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = 1000,
        norm_type: str = "ada_norm_zero",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        action_embed_dim: int = 1024,
        num_in_frames: int = 5,
        num_out_frames: int = 1,
        conv_out_kernel: int = 3,
        patchified_input: bool = False,
        num_patches: Optional[int] = None,
        template_mesh_path: str = None,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.patchified_input = patchified_input
        self.num_in_frames = num_in_frames
        self.num_out_frames = num_out_frames
        
        # 1. Define input layers
        if not patchified_input:
            self.height = sample_size
            self.width = sample_size

        interpolation_scale = self.config.sample_size // 64
        interpolation_scale = max(interpolation_scale, 1)
        if not patchified_input:
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                interpolation_scale=interpolation_scale,
            )
        else:
            assert patch_size is not None and num_patches is not None

            with open(template_mesh_path, 'rb') as file:
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

        # 3. Define temporal transformers blocks
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=None,
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

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        # self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        if not patchified_input:
            self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)
        else:
            # self.proj_out = nn.Linear(inner_dim, patch_size * self.out_channels)
            self.proj_out = nn.Sequential(
                nn.Linear(inner_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, self.out_channels),
            )

        # 5. Latte other blocks.
        # self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)

        self.action_embedding = ActionEmbedding(action_embed_dim)
        
        # define temporal positional embedding
        temp_pos_embed = get_1d_sincos_pos_embed_from_grid(
            inner_dim, torch.arange(0, num_in_frames).unsqueeze(1)
        )  # 1152 hidden size
        self.register_buffer("temp_pos_embed", torch.from_numpy(temp_pos_embed).float().unsqueeze(0), persistent=False)

        if not patchified_input:
            conv_out_padding = (conv_out_kernel - 1) // 2

            self.conv_out = nn.Conv3d(
                in_channels=out_channels,  
                out_channels=out_channels, 
                kernel_size=(num_in_frames, conv_out_kernel, conv_out_kernel), 
                stride=(num_out_frames, 1, 1),
                padding=(num_out_frames - 1, conv_out_padding, conv_out_padding)
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
        # Reshape hidden states
        if not self.patchified_input:
            batch_size, channels, num_frame, height, width = hidden_states.shape
            # batch_size channels num_frame height width -> (batch_size * num_frame) channels height width
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)

            # Input
            height, width = (
                hidden_states.shape[-2] // self.config.patch_size,
                hidden_states.shape[-1] // self.config.patch_size,
            )
            num_patches = height * width
        else:
            if hidden_states.ndim == 5:
                batch_size, channels, num_frame, num_patches, patch_size = hidden_states.shape
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(-1, channels, num_patches, patch_size)
            elif hidden_states.ndim == 4:
                batch_size, num_frame, num_points, num_channels = hidden_states.shape
                hidden_states = hidden_states.reshape(-1, num_points, num_channels)
                

        # input_hidden_states = hidden_states
        hidden_states, input_hidden_states = self.pos_embed(hidden_states)  # already add positional embeddings
        num_patches = hidden_states.shape[1]
        
        # Prepare action embeddings for spatial block
        # batch_size num_tokens hidden_size -> (batch_size * num_frame) num_tokens hidden_size
        encoder_hidden_states = self.action_embedding(encoder_hidden_states)  # batch_size, num_action, 1024
        encoder_hidden_states_spatial = encoder_hidden_states.repeat_interleave(num_frame, dim=0).view(
            -1, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1]
        )

        # for inference only
        if len(timestep.shape) < 1:
            timestep = timestep.expand(batch_size)
        
        # Prepare timesteps for spatial and temporal block
        timestep_spatial = timestep.repeat_interleave(num_frame, dim=0)
        timestep_temp = timestep.repeat_interleave(num_patches, dim=0)

        # Spatial and temporal transformer blocks
        for i, (spatial_block, temp_block) in enumerate(
            zip(self.transformer_blocks, self.temporal_transformer_blocks)
        ):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    spatial_block,
                    hidden_states,
                    None,  # attention_mask
                    encoder_hidden_states_spatial,   # encoder_hidden_states, remember to modify
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
                    encoder_hidden_states_spatial, 
                    encoder_attention_mask,
                    timestep_spatial,
                    None,  # cross_attention_kwargs
                    None,  # class_labels
                )

            if enable_temporal_attentions:
                # (batch_size * num_frame) num_tokens hidden_size -> (batch_size * num_tokens) num_frame hidden_size
                hidden_states = hidden_states.reshape(
                    batch_size, -1, hidden_states.shape[-2], hidden_states.shape[-1]
                ).permute(0, 2, 1, 3)
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])

                if i == 0 and num_frame > 1:
                    hidden_states = hidden_states + self.temp_pos_embed

                if self.training and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        temp_block,
                        hidden_states,
                        None,  # attention_mask
                        None,  # encoder_hidden_states
                        None,  # encoder_attention_mask
                        timestep_temp,
                        None,  # cross_attention_kwargs
                        None,  # class_labels
                        use_reentrant=False,
                    )
                else:
                    hidden_states = temp_block(
                        hidden_states,
                        None,  # attention_mask
                        None,  # encoder_hidden_states
                        None,  # encoder_attention_mask
                        timestep_temp,
                        None,  # cross_attention_kwargs
                        None,  # class_labels
                    )

                # (batch_size * num_tokens) num_frame hidden_size -> (batch_size * num_frame) num_tokens hidden_size
                hidden_states = hidden_states.reshape(
                    batch_size, -1, hidden_states.shape[-2], hidden_states.shape[-1]
                ).permute(0, 2, 1, 3)
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])


        hidden_states = self.norm_out(hidden_states)

        
        if self.patchified_input:
            hidden_states = self.interpolate_features(
                hidden_states, 
                self.interp_index.repeat(hidden_states.size(0), 1, 1).cuda(), 
                self.interp_weight.repeat(hidden_states.size(0), 1, 1).cuda(),
            )
            
            # hidden_states = torch.cat([hidden_states, input_hidden_states], dim=-1)
            hidden_states = hidden_states + input_hidden_states[:, self.inv_patch_index, :]

        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        if not self.patchified_input:
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
            )
            output = output.reshape(
                batch_size, -1, output.shape[-3], output.shape[-2], output.shape[-1]
            ).permute(0, 2, 1, 3, 4)
            output = self.conv_out(output).permute(0, 2, 1, 3, 4)
        else:
            hidden_states = hidden_states.reshape(batch_size, -1, *hidden_states.shape[-2:])
            output = hidden_states[:, -self.num_out_frames:, ...]

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
