import numpy as np
from scipy.ndimage import binary_closing

from lasso_3d.lasso_rotate_vol import create_2D_mask_from_polygon
from lasso_3d.lasso_utils import (
    compute_normal_vector,
    rotate_polygon_to_xy_plane,
)


def cropped_closing(volume, iterations=1):
    """
    Perform binary closing on a cropped version of the volume.

    The volume is cropped to the smallest bounding box around the object.
    """
    # find bounding box
    coords = np.argwhere(volume)
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)

    # crop volume
    volume_cropped = volume[
        min_coords[0] : max_coords[0] + 1,
        min_coords[1] : max_coords[1] + 1,
        min_coords[2] : max_coords[2] + 1,
    ]
    # perform binary closing
    volume_cropped = binary_closing(volume_cropped, iterations=iterations)

    # insert cropped volume into original volume
    volume[
        min_coords[0] : max_coords[0] + 1,
        min_coords[1] : max_coords[1] + 1,
        min_coords[2] : max_coords[2] + 1,
    ] = volume_cropped

    return volume


def get_rounding_permutations(coords, permutation_nr):
    """
    Get all rounding permutations for the given coordinates.

    permutation from 0 to 7:
    0: floor(x), floor(y), floor(z)
    1: floor(x), floor(y), ceil(z)
    2: floor(x), ceil(y), floor(z)
    3: floor(x), ceil(y), ceil(z)
    4: ceil(x), floor(y), floor(z)
    5: ceil(x), floor(y), ceil(z)
    6: ceil(x), ceil(y), floor(z)
    7: ceil(x), ceil(y), ceil(z)
    """
    if permutation_nr & 1:
        coords[:, 2] = np.ceil(coords[:, 2])
    else:
        coords[:, 2] = np.floor(coords[:, 2])

    if permutation_nr & 2:
        coords[:, 1] = np.ceil(coords[:, 1])
    else:
        coords[:, 1] = np.floor(coords[:, 1])

    if permutation_nr & 4:
        coords[:, 0] = np.ceil(coords[:, 0])
    else:
        coords[:, 0] = np.floor(coords[:, 0])

    return coords.astype(int)


def mask_via_extension(polygon_3d, tomo_shape):
    """
    Create a mask by adding slices of the polygon along its normal.

    Steps:
    1. Rotate the polygon to the xy plane.
    2. Create a 2D mask from the rotated polygon.
    3. Rotate the 2D mask back to the original orientation.
    4. Do that for all slices along the z-axis.
    5. Fill holes which appeared during the process.
    """
    # rotate polygon to be flat
    polygon_3d_rotated, polygon_center, rot_mat = rotate_polygon_to_xy_plane(
        polygon_3d.copy()
    )
    normal_vector = compute_normal_vector(polygon_3d)

    polygon_2d = polygon_3d_rotated[:, :2]
    z_component = polygon_3d_rotated[0, 2]

    # create 2D mask
    mask_2d, shift = create_2D_mask_from_polygon(
        polygon_2d.copy(), tomo_shape[:2]
    )

    # get 2D mask coordinates and shift to projected polygon center and add z-component
    mask_coords = np.argwhere(mask_2d)
    mask_coords = np.array(mask_coords, dtype=float)
    mask_coords -= shift * 0.5
    mask_coords_orig = np.concatenate(
        [mask_coords, np.ones((mask_coords.shape[0], 1)) * z_component], axis=1
    )

    volume = np.zeros(tomo_shape, dtype=bool)
    mask_coords_3D = np.dot(mask_coords_orig, rot_mat) + polygon_center
    max_range = np.max(tomo_shape) * 2

    for z in range(-1, -max_range, -1):
        cur_coords = mask_coords_3D + z * normal_vector
        if (
            all(cur_coords[:, 2] < 0)
            or all(cur_coords[:, 2] >= tomo_shape[2])
            or all(cur_coords[:, 0] < 0)
            or all(cur_coords[:, 0] >= tomo_shape[0])
            or all(cur_coords[:, 1] < 0)
            or all(cur_coords[:, 1] >= tomo_shape[1])
        ):
            break
        cur_coords = cur_coords.astype(int)
        cur_coords = cur_coords[
            (cur_coords >= 0).all(axis=1)
            & (cur_coords < tomo_shape).all(axis=1)
        ]
        volume[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = True

    for z in range(max_range):
        cur_coords = mask_coords_3D + z * normal_vector
        if (
            all(cur_coords[:, 2] < 0)
            or all(cur_coords[:, 2] >= tomo_shape[2])
            or all(cur_coords[:, 0] < 0)
            or all(cur_coords[:, 0] >= tomo_shape[0])
            or all(cur_coords[:, 1] < 0)
            or all(cur_coords[:, 1] >= tomo_shape[1])
        ):
            break
        cur_coords = cur_coords.astype(int)
        cur_coords = cur_coords[
            (cur_coords >= 0).all(axis=1)
            & (cur_coords < tomo_shape).all(axis=1)
        ]
        volume[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = True
    # volume1 = binary_closing(volume1)
    volume = cropped_closing(volume)
    return volume
