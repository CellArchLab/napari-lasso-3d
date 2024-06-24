import numpy as np
import open3d as o3d
from scipy.ndimage import binary_fill_holes
from scipy.spatial import Delaunay

from lasso_3d.lasso_utils import (
    compute_normal_vector,
    roll_or_concat,
    rotate_polygon_to_xy_plane,
)


def convert_voxelgrid_to_array(voxel_grid):
    # Convert the voxel grid to a numpy array
    voxel_indices = np.asarray(
        [voxel.grid_index for voxel in voxel_grid.get_voxels()]
    )
    voxel_indices = voxel_indices.astype(np.int32)

    # Get the dimensions of the voxel grid
    min_bound = np.min(voxel_indices, axis=0)
    max_bound = np.max(voxel_indices, axis=0)
    dims = max_bound - min_bound + 1

    # Create an empty numpy array to store the voxel grid
    voxel_array = np.zeros(dims, dtype=np.int8)

    # Fill the numpy array with the voxel data
    for idx in voxel_indices:
        voxel_array[tuple(idx - min_bound)] = 1

    return voxel_array


def find_polygon_distances(polygon_3d, tomo_shape, normal_vector):
    # find first point along normal vector s.t. all shifted points are outside the tomo shape

    t = 0
    while True:
        x_comps = polygon_3d[:, 0] + t * normal_vector[0]
        y_comps = polygon_3d[:, 1] + t * normal_vector[1]
        z_comps = polygon_3d[:, 2] + t * normal_vector[2]

        x_masks = (x_comps < 0) | (x_comps >= tomo_shape[0])
        y_masks = (y_comps < 0) | (y_comps >= tomo_shape[1])
        z_masks = (z_comps < 0) | (z_comps >= tomo_shape[2])

        sample_mask = x_masks | y_masks | z_masks
        if sample_mask.all():
            break

        t += 1
    t_forward = t

    t = 0
    while True:
        x_comps = polygon_3d[:, 0] + t * normal_vector[0]
        y_comps = polygon_3d[:, 1] + t * normal_vector[1]
        z_comps = polygon_3d[:, 2] + t * normal_vector[2]

        x_masks = (x_comps < 0) | (x_comps >= tomo_shape[0])
        y_masks = (y_comps < 0) | (y_comps >= tomo_shape[1])
        z_masks = (z_comps < 0) | (z_comps >= tomo_shape[2])

        sample_mask = x_masks | y_masks | z_masks
        if sample_mask.all():
            break

        t -= 1
    t_backward = t

    return t_forward, t_backward


def get_boundary_vertices(polygon_3d, tomo_shape, normal_vector):
    distance_forward, distance_backward = find_polygon_distances(
        polygon_3d, tomo_shape, normal_vector
    )

    polygon_3d_shifted_forward = (
        polygon_3d.copy() + distance_forward * normal_vector
    )
    polygon_3d_shifted_backward = (
        polygon_3d.copy() + distance_backward * normal_vector
    )

    return polygon_3d_shifted_forward, polygon_3d_shifted_backward


def get_min_shift(polygon_3d_shifted_forward, polygon_3d_shifted_backward):
    """
    This function accounts for the shifts done by voxelizing the mesh.
    """
    min_shift_forward = np.min(polygon_3d_shifted_forward, axis=0)
    min_shift_backward = np.min(polygon_3d_shifted_backward, axis=0)
    min_shift_min = np.min([min_shift_forward, min_shift_backward], axis=0)

    return np.array(min_shift_min, dtype=int)


def get_connecting_faces(polygon_3d, faces_top):
    n = polygon_3d.shape[0]
    faces = []
    # Side faces
    for i in range(n):
        faces.append([i, (i + 1) % n, (i + 1) % n + n])
        faces.append([i, (i + 1) % n + n, i + n])

    faces = np.array(faces)
    faces = np.concatenate((faces, faces_top), axis=0)
    faces = np.concatenate((faces, faces_top + polygon_3d.shape[0]), axis=0)

    return faces


def voxelize_mesh(vertices, faces, tomo_shape):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh, voxel_size=1
    )
    volume = convert_voxelgrid_to_array(voxel_grid)
    volume = binary_fill_holes(volume)

    return volume


def create_volume_from_polygon_mesh(polygon_3d, tomo_shape):
    """
    Create a 3D volumetric mask by voxelizing a 3D polygon mesh.

    Steps:
    1. Rotate the polygon to the xy plane and triangulate it (easy in 2D)
    2. Create a surface surrounding the entire "cone" of the polygon
    3. Voxelize the surface and fill the inside
    4. Roll the volume to the original position
    """
    # rotate the polygon to the xy plane
    polygon_3d_rotated, _, _ = rotate_polygon_to_xy_plane(polygon_3d.copy())
    normal_vector = compute_normal_vector(polygon_3d)

    # create a 2D mask of the rotated polygon
    polygon_2d = polygon_3d_rotated[:, :2]

    # describe polygon by a surface (easiest in 2D)
    tri = Delaunay(polygon_2d)
    faces_top = tri.simplices

    # shift the polygon to the boundary of the tomo shape to cover the whole volume
    polygon_3d_shifted_forward, polygon_3d_shifted_backward = (
        get_boundary_vertices(polygon_3d, tomo_shape, normal_vector)
    )

    # these are shifts introduced by the voxelization
    min_shift_min = get_min_shift(
        polygon_3d_shifted_forward, polygon_3d_shifted_backward
    )

    # stack front and back vertices to cover the entire "cone"
    vertices = np.vstack(
        [polygon_3d_shifted_forward, polygon_3d_shifted_backward]
    )

    # get faces connecting the front and back vertices
    faces = get_connecting_faces(polygon_3d, faces_top)

    # create volumetric mask along the surface borders and fill the inside
    volume = voxelize_mesh(vertices, faces, tomo_shape)

    # roll the volume to the original position
    volume = roll_or_concat(volume, min_shift_min[0], dimension=0)
    volume = roll_or_concat(volume, min_shift_min[1], dimension=1)
    volume = roll_or_concat(volume, min_shift_min[2], dimension=2)
    return volume
