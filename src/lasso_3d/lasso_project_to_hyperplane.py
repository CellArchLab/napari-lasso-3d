import numpy as np
from skimage.draw import polygon2mask

from lasso_3d.lasso_utils import (
    compute_normal_vector,
    rotate_polygon_to_xy_plane,
)


def project_points_to_hyperplane(points, point_on_hyperplane, normal_vector):
    """
    Project points onto a hyperplane defined by a point on the hyperplane and a normal vector.

    This is the bottleneck in terms of performance.
    """
    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Calculate the vector from the point on the hyperplane to the points
    v = points - point_on_hyperplane

    # Calculate the distance from each point to the hyperplane
    d = np.dot(v, normal_vector)

    # Project the points onto the hyperplane
    projected_points = points - np.outer(d, normal_vector)

    return projected_points


def mask_via_projection(polygon_3d, tomo_shape):
    """
    Create a mask by projecting all tomogram coordinates onto the polygon hyperplane
    and checking if they are inside the polygon.

    Steps:
    1. Rotate the polygon to the xy plane, remove its z-component.
    2. Create a 2D mask from the rotated polygon.
    3. Project all tomogram coordinates onto the polygon hyperplane, rotate them with the same
        rotation matrix as the polygon, and project them onto the 2D plane.
    4. Check if the projected coordinates are inside the polygon mask.
    5. Transform the boolean mask to the original shape.

    """
    # Step 1
    # rotate polygon to xy plane
    normal_vector = compute_normal_vector(polygon_3d)
    rotated_polygon, mean_polygon, rotation_matrix = (
        rotate_polygon_to_xy_plane(polygon_3d)
    )
    # # compute normal vector
    # normal_vector = compute_normal_vector(polygon_3d)

    # # subtract mean
    # mean_polygon = np.mean(polygon_3d, axis=0)
    # polygon_3d -= mean_polygon

    # # compute rotation matrix
    # rotation_matrix = rotation_matrix_from_vectors(normal_vector, [0, 0, 1])

    # # rotate polygon
    # rotated_polygon = np.dot(polygon_3d, rotation_matrix.T)

    # project polygon to 2D and center it
    polygon_2d = rotated_polygon[:, :2]
    polygon_2d += np.array(tomo_shape[:2]) // 2

    # Step 2
    # generate mask (2D)
    poly_mask = polygon2mask(tomo_shape[:2], polygon_2d)

    # Step 3
    # get all 3D coordinates of the tomogram
    coords = np.indices(tomo_shape).reshape(3, -1).T

    # project all coordinates onto the polygon hyperplane and subtract the polygon mean
    projected_coords = project_points_to_hyperplane(
        coords, polygon_3d[0], normal_vector
    )
    projected_coords -= mean_polygon

    # rotate all coordinates
    rotated_coords = np.dot(projected_coords, rotation_matrix.T)

    # project all coordinates onto the 2D plane and center them
    rotated_coords = rotated_coords[:, :2] + np.array(tomo_shape[:2]) // 2

    # Step 4
    # convert to integer
    rotated_coords = np.round(rotated_coords).astype(int)

    # exclude out-of-bounds coordinates
    valid_coords = np.all(
        (rotated_coords >= 0) & (rotated_coords < np.array(tomo_shape[:2])),
        axis=1,
    )

    # check for valid coordinates their values in poly_mask
    valid_coords_values = poly_mask[
        rotated_coords[valid_coords][:, 0], rotated_coords[valid_coords][:, 1]
    ]

    # get mask for rotated_coords (shape N, 3)
    in_values = np.zeros(rotated_coords.shape[0], dtype=bool)
    in_values[valid_coords] = valid_coords_values

    # Step 5
    # reshape to 3D
    in_values = in_values.reshape(tomo_shape, order="A")
    return in_values
