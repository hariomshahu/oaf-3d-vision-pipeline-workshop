# %% [markdown]
# # Triangulation
#
# This function triangulates 3D points from two sets of two undistorted normalized
# pixels and a [`TransformationMatrix`](transformation_matrix.py) object. The process
# for this was discussed in more detail in the workshop
# [5: Dual Camera Setups](../workshops/05_dual_camera_setups.ipynb).


# %%
import numpy as np
from nptyping import Float32, NDArray, Shape

from oaf_vision_3d.lens_model import LensModel
from oaf_vision_3d.transformation_matrix import TransformationMatrix


def triangulate_points(
    undistorted_normalized_pixels_0: NDArray[Shape["H, W, 2"], Float32],
    undistorted_normalized_pixels_1: NDArray[Shape["H, W, 2"], Float32],
    transformation_matrix: TransformationMatrix,
) -> NDArray[Shape["H, W, 3"], Float32]:

    # Create camera vectors (add z=1 to make them 3D)
    v0 = np.pad(
        undistorted_normalized_pixels_0, ((0, 0), (0, 0), (0, 1)), constant_values=1.0
    )
    u1 = np.pad(
        undistorted_normalized_pixels_1, ((0, 0), (0, 0), (0, 1)), constant_values=1.0
    )

    # Transform v1 using the transformation matrix
    v1 = transformation_matrix.rotate(u1)

    # Camera positions
    P0 = np.zeros(3, dtype=np.float32)  # Camera 0 at origin
    P1 = transformation_matrix.translation  # Camera 1 position

    # Calculate dot products
    a = np.sum(v0 * v0, axis=-1)  # v0 · v0
    b = np.sum(v0 * v1, axis=-1)  # v0 · v1
    c = np.sum(v1 * v1, axis=-1)  # v1 · v1

    # Calculate (P1 - P0)
    P1_minus_P0 = P1 - P0  # This is just P1 since P0 is at origin

    d = np.sum(v0 * P1_minus_P0[None, None, :], axis=-1)  # v0 · (P1 - P0)
    e = np.sum(v1 * P1_minus_P0[None, None, :], axis=-1)  # v1 · (P1 - P0)

    # Solve for t using the formula: t = (be - cd) / (b² - ac)
    denominator = b * b - a * c

    # Avoid division by zero
    denominator = np.where(np.abs(denominator) < 1e-8, np.nan, denominator)

    t = (b * e - c * d) / denominator

    # Calculate 3D point: P = P0 + t * v0 = t * v0 (since P0 is at origin)
    points_3d = v0 * t[..., None]

    return points_3d


def triangulate_disparity(
    disparity: NDArray[Shape["H, W"], Float32],
    lens_model_0: LensModel,
    lens_model_1: LensModel,
    transformation_matrix: TransformationMatrix,
) -> NDArray[Shape["H, W, 3"], Float32]:
    y, x = np.indices(disparity.shape, dtype=np.float32)
    pixels_0 = np.stack([x, y], axis=-1)
    pixels_1 = np.stack([x - disparity, y], axis=-1)

    undistortied_normalized_pixels_0 = lens_model_0.undistort_pixels(
        normalized_pixels=lens_model_0.normalize_pixels(pixels=pixels_0)
    )
    undistortied_normalized_pixels_1 = lens_model_1.undistort_pixels(
        normalized_pixels=lens_model_1.normalize_pixels(pixels=pixels_1)
    )

    return triangulate_points(
        undistorted_normalized_pixels_0=undistortied_normalized_pixels_0,
        undistorted_normalized_pixels_1=undistortied_normalized_pixels_1,
        transformation_matrix=transformation_matrix,
    )
