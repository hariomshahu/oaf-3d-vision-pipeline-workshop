# %% [markdown]
# # Undistort Image
#
# This function undistorts an image using a new camera matrix as done in the
# [workshop](../workshops/03_image_distortion_and_undistortion.ipynb).

import numpy as np

# %%
from nptyping import Float32, NDArray, Shape
from scipy.ndimage import map_coordinates

from oaf_vision_3d.lens_model import CameraMatrix, LensModel

# def undistort_image_with_new_camera_matrix(  # type: ignore
#     image: NDArray[Shape["H, W, 3"], Float32],
#     lens_model: LensModel,
#     new_camera_matrix: CameraMatrix,
# ) -> NDArray[Shape["H, W, 3"], Float32]: ...


def undistort_image_with_new_camera_matrix(
    image: NDArray[Shape["H, W, 3"], Float32],
    lens_model: LensModel,
    new_camera_matrix: CameraMatrix,
) -> NDArray[Shape["H, W, 3"], Float32]:

    # Create a perfect pixel grid for the output image
    y, x = np.indices(image.shape[:2], dtype=np.float32)
    pixels = np.stack([x, y], axis=-1)

    # This is the key: we normalize using the NEW camera matrix
    # These are the "undistorted" normalized coordinates we want
    normalized_pixels = (
        pixels - np.array([new_camera_matrix.cx, new_camera_matrix.cy])[None, None, :]
    ) / np.array([new_camera_matrix.fx, new_camera_matrix.fy])[None, None, :]

    # Apply distortion to find where these points would be in distorted space
    distorted_normalized_pixels = lens_model.distort_pixels(
        normalized_pixels=normalized_pixels
    )

    # Convert back to pixel coordinates in the original image using original camera matrix
    distorted_pixels = (
        distorted_normalized_pixels
        * np.array([lens_model.camera_matrix.fx, lens_model.camera_matrix.fy])[
            None, None, :
        ]
    ) + np.array([lens_model.camera_matrix.cx, lens_model.camera_matrix.cy])[
        None, None, :
    ]

    # Sample from the original image at these locations
    undistorted_image = np.stack(
        [
            map_coordinates(
                input=_image,
                coordinates=[distorted_pixels[..., 1], distorted_pixels[..., 0]],
                order=1,
                mode="constant",
                cval=0,
            )
            for _image in image.transpose(2, 0, 1)
        ],
        axis=-1,
    )

    return undistorted_image.astype(np.float32)
