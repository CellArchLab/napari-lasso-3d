import numpy as np
from scipy.ndimage import binary_closing

from lasso_3d.lasso_rotate_vol import create_2D_mask_from_polygon
from lasso_3d.lasso_utils import rotate_polygon_to_xy_plane


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

    # fill the volume
    volume = np.zeros(tomo_shape, dtype=bool)
    max_range = np.max(tomo_shape) * 2
    for z in range(-max_range, max_range):
        cur_mask_coords = mask_coords_orig.copy()
        cur_mask_coords[:, 2] += z
        cur_mask_coords = np.dot(cur_mask_coords, rot_mat) + polygon_center
        cur_mask_coords = cur_mask_coords.astype(int)

        # only keep valid coordinates
        cur_mask_coords = cur_mask_coords[
            (cur_mask_coords >= 0).all(axis=1)
            & (cur_mask_coords < tomo_shape).all(axis=1)
        ]

        # fill the volume
        volume[
            cur_mask_coords[:, 0], cur_mask_coords[:, 1], cur_mask_coords[:, 2]
        ] = True

    # fill holes from integer rounding
    volume = binary_closing(volume)
    return volume
