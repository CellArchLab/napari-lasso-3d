import numpy as np


def generate_random_polygon(n_vertices=5, tomo_shape=(100, 100, 100)):
    # create a polygon layer, where all vertices lie on the same plane
    polygon_x = np.random.randint(0, tomo_shape[0], n_vertices)
    polygon_y = np.random.randint(0, tomo_shape[1], n_vertices)
    polygon = np.stack([polygon_x, polygon_y], axis=1)

    # place polygon in 3D space
    polygon_3d = np.zeros((n_vertices, 3))
    polygon_3d[:, 1:] = polygon
    polygon_3d[:, 0] = np.random.randint(10, tomo_shape[0] - 10)

    # rotate the polygon around the tomo center
    polygon_3d = get_rotated_polygon(polygon_3d)
    return polygon_3d


def generate_example_polygon(random_rotation=False):
    # create a polygon layer, where all vertices lie on the same plane
    polygon = np.array([[15, 15], [15, 50], [50, 60], [30, 50], [50, 15]])
    n = polygon.shape[0]

    # place polygon in 3D space
    polygon_3d = np.zeros((n, 3))
    polygon_3d[:, 1:] = polygon
    polygon_3d[:, 0] = 50
    if random_rotation:
        polygon_3d = get_rotated_polygon(polygon_3d)
    return polygon_3d


def rotate_polygon_to_xy_plane(polygon_3d):
    """
    Rotate a 3D polygon s.t. it is parallel to the xy plane.
    """
    polygon_center = np.mean(polygon_3d, axis=0)
    polygon_3d -= polygon_center
    normal_vector = compute_normal_vector(polygon_3d)
    target_vector = np.array([0, 0, 1])
    rot_mat = rotation_matrix_from_vectors(normal_vector, target_vector)
    polygon_3d_rotated = np.dot(polygon_3d, rot_mat.T)
    return polygon_3d_rotated, polygon_center, rot_mat


def get_rotated_polygon(polygon_3d):
    polygon_3d_non_axis_aligned = polygon_3d.copy()

    # rotate polygon around tomo center
    center = np.mean(polygon_3d_non_axis_aligned, axis=0)

    # draw random angle for z-axis rotation, y-axis rotation, and x-axis rotation
    angle_x = np.random.uniform(0, 2 * np.pi)
    angle_y = np.random.uniform(0, 2 * np.pi)
    angle_z = np.random.uniform(0, 2 * np.pi)

    # create rotation matrices
    rotation_matrix_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )
    rotation_matrix_y = np.array(
        [
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ]
    )
    rotation_matrix_z = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1],
        ]
    )

    # rotate the polygon around the center
    polygon_3d_rotated = polygon_3d_non_axis_aligned - center
    polygon_3d_rotated = np.dot(polygon_3d_rotated, rotation_matrix_x.T)
    polygon_3d_rotated = np.dot(polygon_3d_rotated, rotation_matrix_y.T)
    polygon_3d_rotated = np.dot(polygon_3d_rotated, rotation_matrix_z.T)

    # shift the polygon back
    polygon_3d_rotated += center

    # get integer coordinates
    polygon_3d_rotated = np.round(polygon_3d_rotated).astype(float)

    return polygon_3d_rotated


def compute_normal_vector(points):
    # Use the first three points to compute the normal vector
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    return normal


def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def shift_polygon(polygon_2d, max_shape):
    polygon_2d = polygon_2d + np.array(max_shape)

    max_x, max_y = np.max(polygon_2d, axis=0)
    min_x, min_y = np.min(polygon_2d, axis=0)
    shift_array = np.array([max_shape, max_shape])

    if min_x < 0:
        polygon_2d[:, 0] += -min_x
        shift_array[0] = -min_x
    if min_y < 0:
        polygon_2d[:, 1] += -min_y
        shift_array[1] = -min_y
    if max_x >= max_shape * 2:
        polygon_2d[:, 0] -= max_x - max_shape * 2 + 1
        shift_array[0] -= max_x - max_shape * 2 + 1
    if max_y >= max_shape * 2:
        polygon_2d[:, 1] -= max_y - max_shape * 2 + 1
        shift_array[1] -= max_y - max_shape * 2 + 1
    return polygon_2d, shift_array


def roll_or_concat(volume, roll_idcs, dimension):
    if roll_idcs < 0:
        if dimension == 0:
            volume = np.concatenate(
                (np.zeros((abs(roll_idcs),) + volume.shape[1:]), volume),
                axis=dimension,
            )
        elif dimension == 1:
            volume = np.concatenate(
                (
                    np.zeros(
                        (
                            volume.shape[0],
                            abs(roll_idcs),
                        )
                        + volume.shape[2:]
                    ),
                    volume,
                ),
                axis=dimension,
            )
        elif dimension == 2:
            volume = np.concatenate(
                (
                    np.zeros(
                        (volume.shape[0], volume.shape[1], abs(roll_idcs))
                    ),
                    volume,
                ),
                axis=dimension,
            )
        volume = np.roll(volume, roll_idcs, axis=dimension)
        if dimension == 0:
            volume = volume[abs(roll_idcs) :, :, :]
        elif dimension == 1:
            volume = volume[:, abs(roll_idcs) :, :]
        elif dimension == 2:
            volume = volume[:, :, abs(roll_idcs) :]
    else:
        if dimension == 0:
            volume = np.concatenate(
                (volume, np.zeros((roll_idcs,) + volume.shape[1:])),
                axis=dimension,
            )
            volume = np.roll(volume, roll_idcs, axis=dimension)
            volume = volume[:-roll_idcs, :, :]
        elif dimension == 1:
            volume = np.concatenate(
                (
                    volume,
                    np.zeros(
                        (
                            volume.shape[0],
                            roll_idcs,
                        )
                        + volume.shape[2:]
                    ),
                ),
                axis=dimension,
            )
            volume = np.roll(volume, roll_idcs, axis=dimension)
            volume = volume[:, :-roll_idcs, :]
        elif dimension == 2:
            volume = np.concatenate(
                (
                    volume,
                    np.zeros((volume.shape[0], volume.shape[1], roll_idcs)),
                ),
                axis=dimension,
            )
            volume = np.roll(volume, roll_idcs, axis=dimension)
            volume = volume[:, :, :-roll_idcs]
    return volume


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


# def create_volume_from_polygon_mesh(polygon_3d, tomo_shape):
#     # rotate the polygon to the xy plane
#     normal_vector = compute_normal_vector(polygon_3d)
#     target_vector = np.array([0, 0, 1])
#     rot_mat = rotation_matrix_from_vectors(normal_vector, target_vector)
#     polygon_3d_rotated = np.dot(polygon_3d, rot_mat.T)

#     # create a 2D mask of the rotated polygon
#     polygon_2d = polygon_3d_rotated[:, :2]

#     # describe polygon by a surface
#     tri = Delaunay(polygon_2d)
#     faces_top = tri.simplices

#     distance_forward, distance_backward = find_polygon_distances(polygon_3d, tomo_shape, normal_vector)

#     polygon_3d_shifted_forward = polygon_3d.copy() + distance_forward * normal_vector
#     polygon_3d_shifted_backward = polygon_3d.copy() + distance_backward * normal_vector


#     min_shift_forward = np.min(polygon_3d_shifted_forward, axis=0)
#     min_shift_backward = np.min(polygon_3d_shifted_backward, axis=0)
#     min_shift_min = np.min([min_shift_forward, min_shift_backward], axis=0)

#     vertices = np.vstack([polygon_3d_shifted_forward, polygon_3d_shifted_backward])

#     n = polygon_3d.shape[0]
#     faces = []
#     # Side faces
#     for i in range(n):
#         faces.append([i, (i + 1) % n, (i + 1) % n + n])
#         faces.append([i, (i + 1) % n + n, i + n])

#     faces = np.array(faces)
#     faces = np.concatenate((faces, faces_top), axis=0)
#     faces = np.concatenate((faces, faces_top + polygon_3d.shape[0]), axis=0)

#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(vertices)
#     mesh.triangles = o3d.utility.Vector3iVector(faces)
#     voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1)
#     volume = convert_voxelgrid_to_array(voxel_grid)
#     volume = binary_fill_holes(volume)

#     min_idcs = np.array(np.min(polygon_3d, axis=0), dtype=int)
#     min_idcs = np.array(min_shift_min, dtype=int)


#     if min_idcs[0] < 0:
#         volume = np.concatenate((volume, np.zeros((-min_idcs[0], volume.shape[1], volume.shape[2]))), axis=0)
#         volume = np.roll(volume, min_idcs[0], axis=0)
#         volume[min_idcs[0]:, :, :] = 0
#     else:
#         volume = np.concatenate((np.zeros((min_idcs[0], volume.shape[1], volume.shape[2])), volume), axis=0)

#     if min_idcs[1] < 0:
#         volume = np.concatenate((volume, np.zeros((volume.shape[0], -min_idcs[1], volume.shape[2]))), axis=1)
#         volume = np.roll(volume, min_idcs[1], axis=1)
#         volume[:, min_idcs[1]:, :] = 0
#     else:
#         volume = np.concatenate((np.zeros((volume.shape[0], min_idcs[1], volume.shape[2])), volume), axis=1)

#     if min_idcs[2] < 0:
#         volume = np.concatenate((volume, np.zeros((volume.shape[0], volume.shape[1], -min_idcs[2]))), axis=2)
#         volume = np.roll(volume, min_idcs[2], axis=2)
#         volume[:, :, min_idcs[2]:] = 0
#     else:
#         volume = np.concatenate((np.zeros((volume.shape[0], volume.shape[1], min_idcs[2])), volume), axis=2)

#     return  volume
