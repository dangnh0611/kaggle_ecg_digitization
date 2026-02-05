import numpy as np


def batch_perspective_transform_2d(src_pts, M):
    """
    Applies a 2D perspective (homography) transformation to a batch of keypoints.

    This function converts 2D points to homogeneous coordinates, transforms them
    using the provided homography matrices, and performs the perspective division.

    Parameters
    ----------
    src_pts : ndarray of shape (B, N, 2)
        Batch of source keypoints in (x, y) format.
    M : ndarray of shape (B, 3, 3)
        Batch of homography transformation matrices.

    Returns
    -------
    dst_pts : ndarray of shape (B, N, 2)
        Transformed keypoints in (x, y) format after perspective division.

    Notes
    -----
    The transformation follows the projection:
    [x', y', w']^T = M * [x, y, 1]^T
    Final coordinates = (x'/w', y'/w')

    Warning: This function does not explicitly handle cases where the scale factor
    w' is zero (points at infinity or vanishing line).
    """
    B, N, _ = src_pts.shape
    # Create homogeneous coordinates (B, N, 3)
    ones = np.ones((B, N, 1), dtype=src_pts.dtype)
    src_h = np.concatenate([src_pts, ones], axis=-1)

    # Batch matrix multiplication: (B, 3, 3) @ (B, 3, N) -> (B, 3, N)
    # Then transpose back to (B, N, 3)
    dst_h = np.matmul(M, src_h.transpose(0, 2, 1)).transpose(0, 2, 1)

    # Extract the scale factor (w)
    w = dst_h[..., 2:3]

    # Avoid division by zero with a small epsilon
    # or use np.where to keep points at infinity as a specific value
    eps = 1e-8
    w_stable = np.where(np.abs(w) < eps, eps, w)

    return dst_h[..., :2] / w_stable
