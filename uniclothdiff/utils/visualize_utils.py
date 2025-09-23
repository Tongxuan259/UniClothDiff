import trimesh
import numpy as np
import imageio

def init_mesh_from_file(mesh_file):
    mesh = trimesh.load(mesh_file)
    return mesh

def init_mesh_from_primitive(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def get_mesh_image(mesh, camera_angles=None):
    scene = mesh.scene()
    
    if camera_angles is not None:
        scene.set_camera(angles=camera_angles, distance=10, center=mesh.centroid, fov=[60, 40])

    image = scene.save_image(resolution=[1920, 1080], visible=True)
    
    return image

