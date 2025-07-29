# %% [markdown]
# # Solve PnP
#
# This function solves the PnP problem as done in the
# [workshop](../workshops/04_3d_2d_projections_and_pnp.ipynb).

# %%
import cv2
import numpy as np
from nptyping import Float32, NDArray, Shape

from oaf_vision_3d.lens_model import LensModel


def project_points_with_jacobian(
    points: NDArray[Shape["*, 3"], Float32],
    rvec: NDArray[Shape["3"], Float32],
    tvec: NDArray[Shape["3"], Float32],
    lens_model: LensModel,
) -> tuple[NDArray[Shape["*, 2"], Float32], NDArray[Shape["*, 2, 6"], Float32]]:
    projected_points, jacobian = cv2.projectPoints(
        objectPoints=points,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=lens_model.camera_matrix.as_matrix(),
        distCoeffs=lens_model.distortion_coefficients.as_opencv_vector(),
    )
    return (
        projected_points.astype(np.float32)[:, 0, :],
        jacobian.astype(np.float32)[:, :6].reshape(-1, 2, 6),
    )


# def solve_pnp(  # type: ignore
#     points: NDArray[Shape["*, 3"], Float32],
#     pixels: NDArray[Shape["*, 2"], Float32],
#     lens_model: LensModel,
#     rvec: NDArray[Shape["3"], Float32] = np.zeros(3, dtype=np.float32),
#     tvec: NDArray[Shape["3"], Float32] = np.zeros(3, dtype=np.float32),
#     epsilon: float = 1e-5,
#     max_iterations: int = 100,
# ) -> tuple[NDArray[Shape["3"], Float32], NDArray[Shape["3"], Float32]]: ...


def solve_pnp(
    points: NDArray[Shape["*, 3"], Float32],
    pixels: NDArray[Shape["*, 2"], Float32],
    lens_model: LensModel,
    rvec: NDArray[Shape["3"], Float32] = np.zeros(3, dtype=np.float32),
    tvec: NDArray[Shape["3"], Float32] = np.zeros(3, dtype=np.float32),
    epsilon: float = 1e-5,
    max_iterations: int = 100,
) -> tuple[NDArray[Shape["3"], Float32], NDArray[Shape["3"], Float32]]:
    
    # Current estimates
    current_rvec = rvec.copy()
    current_tvec = tvec.copy()
    
    for iteration in range(max_iterations):
        # Project points and get Jacobian
        projected_pixels, jacobian = project_points_with_jacobian(
            points=points,
            rvec=current_rvec,
            tvec=current_tvec,
            lens_model=lens_model,
        )
        
        # Compute error vector (observed - projected)
        error = pixels - projected_pixels  # Shape: (N, 2)
        error_vector = error.reshape(-1)   # Shape: (2N,)
        
        # Reshape Jacobian from (N, 2, 6) to (2N, 6)
        jacobian_matrix = jacobian.reshape(-1, 6)
        
        # Solve the linear system: J^T * J * delta = J^T * error
        # Using least squares: delta = (J^T * J)^-1 * J^T * error
        try:
            delta, residuals, rank, s = np.linalg.lstsq(jacobian_matrix, error_vector, rcond=None)
        except np.linalg.LinAlgError:
            # If singular, return current estimates
            break
        
        # Update estimates
        current_rvec = current_rvec + delta[:3]
        current_tvec = current_tvec + delta[3:6]
        
        # Check convergence
        if np.linalg.norm(delta) < epsilon:
            break
    
    return current_rvec, current_tvec
    