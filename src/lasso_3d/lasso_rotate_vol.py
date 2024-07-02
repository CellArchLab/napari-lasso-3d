import numpy as np
from scipy.ndimage import affine_transform
from skimage.draw import polygon2mask

from lasso_3d.lasso_utils import (
    roll_or_concat,
    rotate_polygon_to_xy_plane,
)


def create_2D_mask_from_polygon(polygon_2d):
    """
    Create a 2D mask from a 2D polygon.
    """
    min_val = np.min(polygon_2d)
    max_val = np.max(polygon_2d)

    polygon_2d += -min_val + 5

    mask_shape = np.array([max_val - min_val + 10, max_val - min_val + 10])
    mask_shape = np.round(mask_shape).astype(int)
    mask = polygon2mask(mask_shape, polygon_2d)
    return mask, -min_val + 5


def extend_2D_mask_to_3D_volume(mask, tomo_shape):
    """
    Extend a 2D mask to a 3D volume by repeating it along the z-axis.
    """
    volume = np.zeros(
        (mask.shape[0], mask.shape[1], max(tomo_shape)), dtype=int
    )
    for z in range(volume.shape[2]):
        volume[:, :, z] = mask
    return volume


def calculate_offset(volume, rot_mat):
    center = np.array(volume.shape) / 2
    offset = -center + np.dot(rot_mat, center)
    return offset


def extend_polygon_to_3D_mask_voxels(polygon_3d, tomo_shape):
    """
    Create a 3D volume from a 3D polygon by extending it along the z-axis.

    How it works:
    1. Rotate the polygon to the xy plane.
    2. Create a 2D mask of the rotated polygon.
    3. Create a 3D volume from the 2D mask by extending it along the z-axis.
    4. Rotate the 3D volume back to the original orientation.

    """

    # rotate the polygon to the xy plane
    polygon_3d_rotated, polygon_center, rot_mat = rotate_polygon_to_xy_plane(
        polygon_3d
    )

    # create a 2D mask of the rotated polygon
    polygon_2d = polygon_3d_rotated[:, :2]
    mask, _ = create_2D_mask_from_polygon(polygon_2d, tomo_shape)

    # create a 3D volume from the 2D mask by stacking it along the z-axis
    volume = extend_2D_mask_to_3D_volume(mask, tomo_shape)

    # rotate the volume
    offset = calculate_offset(volume, rot_mat)
    volume_rotated = affine_transform(volume, rot_mat, order=0, offset=-offset)

    # roll the volume to the original position
    polygon_shift = np.array(volume_rotated.shape) // 2 - polygon_center
    polygon_shift *= -1

    volume_rotated = roll_or_concat(volume_rotated, int(polygon_shift[0]), 0)
    volume_rotated = roll_or_concat(volume_rotated, int(polygon_shift[1]), 1)
    volume_rotated = roll_or_concat(volume_rotated, int(polygon_shift[2]), 2)

    # crop to tomo size
    volume_rotated = volume_rotated[
        : tomo_shape[0], : tomo_shape[1], : tomo_shape[2]
    ]

    return volume_rotated
