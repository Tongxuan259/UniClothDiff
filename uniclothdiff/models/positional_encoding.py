import torch
import torch.nn as nn
import numpy as np
from diffusers.models.embeddings import get_2d_sincos_pos_embed

def fourier_action_embedding(embed_dim, action):
    """
    Args:
        embed_dim: int
        box: a 3-D tensor [B x N x 3] representing the action [A_x, A_y, A_z]
    Returns:
        [B x N x embed_dim] tensor of positional embeddings
    """
    batch_size, num_act_points = action.shape[:2]

    emb = 100 ** (torch.arange(embed_dim) / embed_dim)
    emb = emb[None, None, None].to(device=action.device, dtype=action.dtype)
    emb = emb * action.unsqueeze(-1)

    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
    emb = emb.permute(0, 1, 3, 4, 2).reshape(batch_size, num_act_points, embed_dim * 2 * 3)

    return emb

def get_3d_sincos_pos_embed(embed_dim, patch_centers, scale=1.0):
    """
    Generate fixed 3D sinusoidal positional embeddings for patches based on their center positions.
    
    Args:
    - embed_dim (int): Dimension of the embedding vector. Must be divisible by 6.
    - patch_centers (torch.Tensor): Center points for each patch, shape [N_patches, 3] 
      where N_patches is number of patches (e.g., 100).
    - scale (float): Scaling factor for the embeddings.
    
    Returns:
    - torch.Tensor: Positional embeddings of shape [N_patches, embed_dim]
    """
    # assert embed_dim % 6 == 0, "Embedding dimension must be divisible by 6"
    num_patches, _ = patch_centers.shape
    
    # Normalize patch centers
    patch_centers = patch_centers - np.mean(patch_centers, axis=0)
    scale = np.linalg.norm(patch_centers, axis=1).max()
    patch_centers = patch_centers / scale
    
    # Generate positional embeddings using sine and cosine
    pos_embed = np.zeros((num_patches, embed_dim))
    div_term = np.exp(np.arange(0, embed_dim, 6) * -(np.log(10000.0) / embed_dim))
    for i in range(0, embed_dim, 6):
        pos_embed[:, i:i+3] = np.sin(patch_centers * div_term[i // 6])
        pos_embed[:, i+3:i+6] = np.cos(patch_centers * div_term[i // 6])

    return pos_embed
    


class ActionEmbedding(nn.Module):
    def __init__(self, 
                 out_dim: int, 
                 fourier_freqs: int = 8):
        super().__init__()
        self.out_dim = out_dim

        self.fourier_embedder_dim = fourier_freqs
        self.position_dim = fourier_freqs * 2 * 3  # 2: sin/cos, 4: A_x, A_y, A_z

        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]

        self.linears = nn.Sequential(
            nn.Linear(self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        # self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(
        self,
        actions: torch.Tensor,
    ):
        # embedding position (it may includes padding as placeholder)
        action_embedding = fourier_action_embedding(self.fourier_embedder_dim, actions)  # B*N*3 -> B*N*C
        
        action_embedding = self.linears(action_embedding)

        return action_embedding
    
class PachifiedEmbed(nn.Module):
    """Patchified input embedding"""

    def __init__(
        self,
        in_channels=4,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        pos_embed_type="sincos",
        patch_index=None,
        patch_offset=None,
        patch_centers=None,
    ):
        super().__init__()
        self.flatten = flatten
        self.layer_norm = layer_norm
        
        self.patch_index = patch_index
        self.patch_offset = patch_offset
        self.patch_size = torch.tensor([
            self.patch_offset[i] - self.patch_offset[i-1] \
                for i in range(1, self.patch_offset.shape[0])
            ]).to(self.patch_offset.device)
        self.conv1 =nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        if pos_embed_type is None:
            self.pos_embed = None
        elif pos_embed_type == "sincos":
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=embed_dim,
                patch_centers=patch_centers.squeeze().numpy()
            )
            
            self.register_buffer("pos_embed", 
                                 torch.from_numpy(pos_embed).float().unsqueeze(0), 
                                 persistent=False)
        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

    def forward(self, latent):
        # [batch_size, C, num_patches, patch_size] --> [batch_size, num_patches, patch_size, C]
        # rearrange order of points according to patch order
        latent = latent[:, self.patch_index.int(), :]

        # First PointNet layer
        latent = self.conv1(latent)
        y = torch.cat([
            torch.max(
                latent[:, self.patch_offset[i]:self.patch_offset[i+1], :], 
                dim=1, 
                keepdim=True
            ).values for i in range(self.patch_offset.shape[0] - 1)
        ], dim=1)

        # Concatenate global feature with each point's feature
        latent = torch.cat([
            y.repeat_interleave(
                self.patch_size.cuda(), 
                dim=1
            ),
            latent
        ], dim=-1)
        
        # Second PointNet layer
        latent = self.conv2(latent)
        raw_latent = latent
        latent = torch.cat([
            torch.max(
                latent[:, self.patch_offset[i]:self.patch_offset[i+1], :], 
                dim=1, 
                keepdim=True
            ).values for i in range(self.patch_offset.shape[0] - 1)
        ], dim=1)
        
        # Add positional embedding
        pos_embed = self.pos_embed
        latent = (latent + pos_embed).to(latent.dtype)
        return latent, raw_latent