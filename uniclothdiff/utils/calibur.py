
import torch
import numpy as np

class CameraConventions:
    """
    Conventions of camera coordinate systems in *right, up, forward* order.
    Also available as :py:data:`CC` for faster typing.
    """
    GL = ("X", "Y", "-Z")
    Godot = Blender = OpenGL = GL
    CV = ("X", "-Y", "Z")
    OpenCV = CV
    ROS = ("-Y", "Z", "X")
    DirectXLH = ("X", "Y", "Z")
    Unity = DirectXLH
    UE = ("Y", "Z", "X")
    
CC = CameraConventions

axes = {
     "X": [ 1, 0, 0],
    "-X": [-1, 0, 0],
     "Y": [ 0, 1, 0],
    "-Y": [ 0,-1, 0],
     "Z": [ 0, 0, 1],
    "-Z": [ 0, 0,-1]
}


def convert_pose(src_pose, src_convention, dst_convention):
    basis_a = get_ruf_basis(src_convention)
    basis_b = get_ruf_basis(dst_convention)
    transform = torch.eye(4, dtype=basis_a.dtype, device=src_pose.device)
    transform[:3, :3] = torch.matmul(basis_a, basis_b.t())
    return src_pose @ transform

def get_ruf_basis(convention):
    """
    Get the *right/up/forward* basis matrix in ``convention``.
    The matrix is laid out as ``[r|u|f]``.
    """
    r, u, f = convention
    r, u, f = axes[r], axes[u], axes[f]
    return torch.tensor(np.stack([r, u, f], axis=-1).astype(np.float32))