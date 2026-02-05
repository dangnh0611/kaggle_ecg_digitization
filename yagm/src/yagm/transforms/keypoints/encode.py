import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import chi2

logger = logging.getLogger(__name__)


def _clip(v, maxv, minv=0):
    return min(maxv, max(minv, v))


def generate_simple_3d_gaussian_heatmap(
    heatmap_size: Tuple[int, int, int, int],
    keypoints: torch.Tensor,
    sigma: float | None = None,
) -> torch.Tensor:
    """Generate simple 3D Gaussian heatmap from keypoints.
    This simplify that the covariance matrix are diagonal,
    so the orientation of the ellipsoid are parallel to base axes X, Y, Z.

    Args:
        heatmap_size: heatmap size of shape (C, X, Y, Z) where C is the number of classes
        keypoints: (N, D) where D is one of:
            3 -> (x, y, z)
            4 -> (x, y, z, class)
            5 -> (x, y, z, sigma, class)
            7 -> (x, y, z, sigma_x, sigma_y, sigma_z, class)
        sigma: the Gaussian sigma parameter

    Returns:
        Tensor of shape (C, X, Y, Z) as provided by `heatmap_size`
    """
    C, X, Y, Z = heatmap_size
    if keypoints.shape[1] < 5:  # keypoints not contain sigma information
        if isinstance(sigma, (list, tuple)):
            sigma_x, sigma_y, sigma_z = sigma
        elif sigma is None:
            raise ValueError
        else:
            sigma_x = sigma_y = sigma_z = sigma
    elif sigma is not None:
        raise ValueError

    # create 3D meshgrid
    xs = torch.linspace(0.5, X - 0.5, X)
    ys = torch.linspace(0.5, Y - 0.5, Y)
    zs = torch.linspace(0.5, Z - 0.5, Z)
    # shape (X, Y, Z)
    grid = torch.stack(torch.meshgrid([xs, ys, zs], indexing="ij"), dim=-1)

    heatmap = torch.zeros(heatmap_size, dtype=torch.float32)
    for kpt in keypoints.tolist():
        if len(kpt) == 3:
            x, y, z = kpt
            kpt_cls = 0
        elif len(kpt) == 4:
            x, y, z, kpt_cls = kpt
        elif len(kpt) == 5:
            x, y, z, sigma, kpt_cls = kpt
            sigma_x = sigma_y = sigma_z = sigma
        elif len(kpt) == 7:
            x, y, z, sigma_x, sigma_y, sigma_z, kpt_cls = kpt
        else:
            raise ValueError

        # voxel indices -> floating point coordinate
        # origin is top-left corner of top-left voxel
        # center of first voxel (top-left) has coordinate (0.5, 0.5, 0.5)
        x += 0.5
        y += 0.5
        z += 0.5

        kpt_cls = int(kpt_cls)
        assert kpt_cls < C

        # 4-sigma rule (3D)
        x_min = _clip(round(x - 4 * sigma_x), X)
        x_max = _clip(round(x + 4 * sigma_x), X, x_min + 1)
        y_min = _clip(round(y - 4 * sigma_y), Y)
        y_max = _clip(round(y + 4 * sigma_y), Y, y_min + 1)
        z_min = _clip(round(z - 4 * sigma_z), Z)
        z_max = _clip(round(z + 4 * sigma_z), Z, z_min + 1)
        if (x_max <= x_min) or (y_max <= y_min) or (z_max <= z_min):
            continue

        kpt_tensor = torch.tensor([x, y, z], dtype=torch.float32)[
            None, None, None
        ]  # (1, 1, 1, 3)
        grid_patch = grid[x_min:x_max, y_min:y_max, z_min:z_max]
        # shape (1, 1, 1, 3)
        sigma2_tensor = (
            torch.tensor([sigma_x, sigma_y, sigma_z], dtype=torch.float32)[
                None, None, None
            ]
            ** 2
        )
        gaussian = torch.exp(
            -0.5 * torch.sum(((grid_patch - kpt_tensor) ** 2) / sigma2_tensor, dim=-1)
        )
        heatmap_patch = heatmap[kpt_cls, x_min:x_max, y_min:y_max, z_min:z_max]
        assert gaussian.shape == heatmap_patch.shape
        torch.maximum(heatmap_patch, gaussian, out=heatmap_patch)

    return heatmap


def _check_valid_covariance_matrix(matrix: torch.Tensor):
    """Validates if a tensor is a valid 3x3 covariance matrix.

    Raises:
        TypeError: If the input is not a torch.Tensor.
        ValueError: If the matrix is not 3x3 or is not a valid covariance matrix.
    """

    if not isinstance(matrix, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")
    dof = matrix.shape[0]

    if matrix.shape != (dof, dof):
        raise ValueError(f"Matrix must be {dof}x{dof}, but got shape {matrix.shape}.")

    if not torch.allclose(matrix, matrix.T):
        raise ValueError("Matrix is not symmetric.")

    try:
        torch.linalg.cholesky(matrix)
    except RuntimeError as e:
        if "positive definite" in str(e):
            raise ValueError("Matrix is not positive definite.")
        elif "singular" in str(e):
            print("Matrix is singular (checking eigenvalues for semi-definiteness).")
            eigenvalues = torch.linalg.eigvals(matrix)
            if not torch.all(eigenvalues.real >= 0):
                raise ValueError(
                    "Matrix is not positive semi-definite. Eigenvalues:", eigenvalues
                )
        else:
            raise RuntimeError(
                f"An unexpected error occurred during Cholesky decomposition: {e}"
            )


def cal_prob_outside_conf_interval(d, sigma_scale_factor=None, conf_interval=None):
    """
    Compute the maximum probability at the boundary of the x% confidence region
    for a d-dimensional standard multivariate normal distribution.

    Parameters:
        d (int): Number of dimensions (e.g., 3 for 3D).
        sigma_scale_factor: value of `sigma_scale_factor` sigma-rule,
            e.g 3.0 for dof=1 within 99.73% confident interval
        conf_interval (float): Confidence level in range[0, 1]

    Returns:
        float: Maximum probability density outside the confidence region.
    """
    if sigma_scale_factor is None:
        # Compute the chi-squared quantile (squared Mahalanobis distance)
        squared_maha = chi2.ppf(conf_interval, df=d)
    else:
        assert conf_interval is None
        squared_maha = sigma_scale_factor**2

    # Compute the PDF value at that quantile distance
    prob = np.exp(-0.5 * squared_maha)
    return prob


def cal_range_in_conf_interval(
    cov: torch.Tensor,
    sigma_scale_factor: float | None = None,
    conf_interval: float | None = 0.999,
) -> torch.Tensor:
    """
    Given a normal distribution with covariance matrix `cov`.
    This function calculate value ranges (ellipsoid radius) along base axes X, Y, Z
    lies within a confident inverval, e.g 99.9% as default
    This is a generalized version of 3-sigma rules where `cov` matrix is just 1x1 matrix
    of 1D variance, conf_interval ~ 99.73%, this function should return
    `3 * sigma = 3*sqrt(cov[0,0])`

    Args:
        cov: covariance matrix of shape (D, D) where D is the number of dimensions
        sigma_scale_factor: value of `sigma_scale_factor` sigma-rule, e.g 3.0 for dof=1
        conf_interval: the confident interval

    Returns:
        Tensor R of shape (D,) specify the range along D base axes.
        The region within confident interval then be [mean - R, mean + R]
    """
    dof = cov.shape[0]
    assert cov.shape == (dof, dof)

    if sigma_scale_factor is not None:
        assert conf_interval is None and sigma_scale_factor > 0
    else:
        # chi-squared critical value for K degrees of freedom
        sigma_scale_factor = chi2.ppf(conf_interval, df=dof) ** 0.5

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # faster for symmetric matrices
    assert torch.all(eigenvalues >= 0.0), f"{cov}"
    ax_lengths = torch.sqrt(eigenvalues) * sigma_scale_factor
    ax_vecs = ax_lengths[None] * eigenvectors  # each colume is an axis vector
    ret = ax_vecs.abs().max(dim=-1)[0]
    return ret


def cal_std_along_principle_axes(
    cov: torch.Tensor,
) -> torch.Tensor:
    """
    Given a normal distribution with covariance matrix `cov`.
    This function calculate standard deviation (std) along base axes X, Y, Z

    Args:
        cov: covariance matrix of shape (D, D) where D is the number of dimensions

    Returns:
        Tensor R of shape (D,) specify the range along D base axes.
        The region within confident interval then be [mean - R, mean + R]
    """
    dof = cov.shape[0]
    assert cov.shape == (dof, dof)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # faster for symmetric matrices
    assert torch.all(eigenvalues >= 0.0), f"{cov}"
    ax_lengths = torch.sqrt(eigenvalues)
    return ax_lengths.tolist()


def generate_3d_gaussian_heatmap(
    heatmap_size: Tuple[int, int, int, int],
    keypoints: torch.Tensor,
    stride: Tuple[int, int, int] = 1,
    covariance: torch.Tensor | None = None,
    dtype=torch.float32,
    sigma_scale_factor: float | None = None,
    conf_interval: float | None = 0.999,
    lower=0.0,
    upper=1.0,
    same_std=False,
    add_offset=True,
    validate_cov_mat=False,
) -> torch.Tensor:
    """Generate 3D multivariate Gaussian heatmap from keypoints with arbitrary covariance matrix.

    Args:
        heatmap_size: heatmap size of shape (C, X, Y, Z) where C is the number of classes
        keypoints: (N, D) where D is one of:
            3 -> (x, y, z)
            4 -> (x, y, class)
            10 -> (x, y, z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz, class)
        convariance: (3, 3) covariance matrix (squared of sigma/std) if not provided per keypoint
        dtype: output heatmap data type, use torch.float16 to save some memory else torch.float32
        sigma_scale_factor: number of sigma-multiplicative rule, e.g 3 for dof=1, 4.03 for dof=3
        conf_interval: amount of confident interval drawed, covered by `sigma_scale_factor` sigma rule
        lower: min heatmap value for label smoothing
        upper: max heatmap value for label smoothing
        add_offset: whether to add 0.5 offset to coordinates to transform from discreate pixel indices
            to floating point coordinate
        validate_cov_mat: whether to validate the per-keypoint covariance matrix

    Returns:
        Tensor of shape (C, X, Y, Z) as provided by `heatmap_size`
    """
    assert 0 <= lower < upper <= 1.0
    assert len(keypoints.shape) == 2 and keypoints.shape[1] >= 3
    keypoints = keypoints.clone()
    # voxel indices -> floating point coordinate
    # origin is top-left corner of top-left voxel
    # center of first voxel (top-left) has coordinate (0.5, 0.5, 0.5)
    if add_offset:
        keypoints[:, :3] += 0.5

    C = heatmap_size[0]
    assert all([e % s == 0 for e, s in zip(heatmap_size[1:], stride)])
    X, Y, Z = [round(e / s) for e, s in zip(heatmap_size[1:], stride)]

    # Create 3D grid
    xs = torch.linspace(0.5, X - 0.5, X)
    ys = torch.linspace(0.5, Y - 0.5, Y)
    zs = torch.linspace(0.5, Z - 0.5, Z)
    grid = torch.stack(
        torch.meshgrid([xs, ys, zs], indexing="ij"), dim=-1
    )  # Shape (X, Y, Z, 3)

    heatmap = torch.full((C, X, Y, Z), lower, dtype=dtype)

    stride_tensor = torch.tensor([stride], dtype=torch.float32)  # (1, 3)
    if keypoints.shape[1] <= 4:
        keypoints[:, :3] /= stride_tensor
        assert covariance.shape == (3, 3)
    elif keypoints.shape[1] == 10:
        keypoints[:, :3] /= stride_tensor
        # cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz
        keypoints[:, 3] /= stride[0] * stride[0]
        keypoints[:, 4] /= stride[1] * stride[1]
        keypoints[:, 5] /= stride[2] * stride[2]
        keypoints[:, 6] /= stride[0] * stride[1]
        keypoints[:, 7] /= stride[0] * stride[2]
        keypoints[:, 8] /= stride[1] * stride[2]
    else:
        raise ValueError

    if sigma_scale_factor is not None:
        assert conf_interval is None and sigma_scale_factor > 0
    else:
        # chi-squared critical value for K degrees of freedom
        sigma_scale_factor = chi2.ppf(conf_interval, df=3) ** 0.5

    for kpt in keypoints.tolist():
        if len(kpt) == 3:
            x, y, z = kpt
            kpt_cls = 0
            cov_mat = covariance
        elif len(kpt) == 4:
            x, y, z, kpt_cls = kpt
            cov_mat = covariance
        elif len(kpt) == 10:
            x, y, z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz, kpt_cls = kpt
            cov_mat = torch.tensor(
                [
                    [cov_xx, cov_xy, cov_xz],
                    [cov_xy, cov_yy, cov_yz],
                    [cov_xz, cov_yz, cov_zz],
                ],
                dtype=torch.float32,
            )
        else:
            raise ValueError

        if validate_cov_mat:
            _check_valid_covariance_matrix(cov_mat)

        kpt_cls = int(kpt_cls)
        assert kpt_cls < C

        if not same_std:
            # compute values interval along base axes XYZ, inside a confident inverval
            # e.g 99.9% <-> 4.0331422236561565 sigma-rule
            range_x, range_y, range_z = cal_range_in_conf_interval(
                cov_mat,
                conf_interval=None,
                sigma_scale_factor=sigma_scale_factor,
            ).tolist()
            assert range_x > 0 and range_y > 0 and range_z > 0
        else:
            principle_stds = cal_std_along_principle_axes(cov_mat)
            std = min(principle_stds)
            assert std > 0
            var = std**2
            cov_mat = torch.tensor(
                [
                    [var, 0, 0],
                    [0, var, 0],
                    [0, 0, var],
                ],
                dtype=torch.float32,
            )
            range_x = range_y = range_z = std * sigma_scale_factor

        # slices to crop -> reduce computation
        x_min = _clip(round(x - range_x), X)
        x_max = _clip(round(x + range_x), X, x_min + 1)
        y_min = _clip(round(y - range_y), Y)
        y_max = _clip(round(y + range_y), Y, y_min + 1)
        z_min = _clip(round(z - range_z), Z)
        z_max = _clip(round(z + range_z), Z, z_min + 1)
        if (x_max <= x_min) or (y_max <= y_min) or (z_max <= z_min):
            continue

        # Precompute covariance matrix inverse and determinant
        sigma_inv = torch.linalg.inv(cov_mat)
        grid_patch = grid[x_min:x_max, y_min:y_max, z_min:z_max]
        grid_diff = grid_patch - torch.tensor(
            [[[[x, y, z]]]], dtype=dtype
        )  # Shape (X, Y, Z, 3) - [1,1,1,3]

        # Compute multivariate Gaussian
        squared_mahalanobis = torch.einsum(
            "...i,ij,...j->...", grid_diff, sigma_inv, grid_diff
        )  # Shape (X, Y, Z)
        gaussian = torch.exp(-0.5 * squared_mahalanobis)  # min~0, max~1
        # turn into [0,1] range
        gaussian = (gaussian - gaussian.min()) / (
            gaussian.max() - gaussian.min()
        )  # min=0, max=1
        gaussian = lower + (upper - lower) * gaussian  # min=lower, max=upper

        heatmap_patch = heatmap[kpt_cls, x_min:x_max, y_min:y_max, z_min:z_max]
        torch.maximum(heatmap_patch, gaussian, out=heatmap_patch)

    return heatmap


def generate_3d_segment_mask(
    heatmap_size: Tuple[int, int, int, int],
    keypoints: torch.Tensor,
    stride: Tuple[int, int, int] = 1,
    covariance: torch.Tensor | None = None,
    dtype=torch.float32,
    sigma_scale_factor: float | None = None,
    conf_interval: float | None = 0.999,
    lower=0.0,
    upper=1.0,
    same_std=False,
    add_offset=True,
    validate_cov_mat=False,
) -> torch.Tensor:
    """Generate 3D segmentation map from keypoints with arbitrary covariance matrix.

    Args:
        heatmap_size: heatmap size of shape (C, X, Y, Z) where C is the number of classes
        keypoints: (N, D) where D is one of:
            3 -> (x, y, z)
            4 -> (x, y, class)
            10 -> (x, y, z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz, class)
        convariance: (3, 3) covariance matrix (squared of sigma/std) if not provided per keypoint
        dtype: output heatmap data type, use torch.float16 to save some memory else torch.float32
        sigma_scale_factor: number of sigma-multiplicative rule, e.g 3 for dof=1, 4.03 for dof=3
        conf_interval: amount of confident interval drawed, covered by `sigma_scale_factor` sigma rule
        lower: min heatmap value for label smoothing
        upper: max heatmap value for label smoothing
        add_offset: whether to add 0.5 offset to coordinates to transform from discreate pixel indices
            to floating point coordinate
        validate_cov_mat: whether to validate the per-keypoint covariance matrix

    Returns:
        Tensor of shape (C, X, Y, Z) as provided by `heatmap_size`
    """
    assert 0 <= lower < upper <= 1.0
    assert len(keypoints.shape) == 2 and keypoints.shape[1] >= 3
    keypoints = keypoints.clone()
    # voxel indices -> floating point coordinate
    # origin is top-left corner of top-left voxel
    # center of first voxel (top-left) has coordinate (0.5, 0.5, 0.5)
    if add_offset:
        keypoints[:, :3] += 0.5

    C = heatmap_size[0]
    assert all([e % s == 0 for e, s in zip(heatmap_size[1:], stride)])
    X, Y, Z = [round(e / s) for e, s in zip(heatmap_size[1:], stride)]

    # Create 3D grid
    xs = torch.linspace(0.5, X - 0.5, X)
    ys = torch.linspace(0.5, Y - 0.5, Y)
    zs = torch.linspace(0.5, Z - 0.5, Z)
    grid = torch.stack(
        torch.meshgrid([xs, ys, zs], indexing="ij"), dim=-1
    )  # Shape (X, Y, Z, 3)

    heatmap = torch.full((C, X, Y, Z), lower, dtype=dtype)

    stride_tensor = torch.tensor([stride], dtype=torch.float32)  # (1, 3)
    if keypoints.shape[1] <= 4:
        keypoints[:, :3] /= stride_tensor
        assert covariance.shape == (3, 3)
    elif keypoints.shape[1] == 10:
        keypoints[:, :3] /= stride_tensor
        # cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz
        keypoints[:, 3] /= stride[0] * stride[0]
        keypoints[:, 4] /= stride[1] * stride[1]
        keypoints[:, 5] /= stride[2] * stride[2]
        keypoints[:, 6] /= stride[0] * stride[1]
        keypoints[:, 7] /= stride[0] * stride[2]
        keypoints[:, 8] /= stride[1] * stride[2]
    else:
        raise ValueError

    if sigma_scale_factor is not None:
        assert conf_interval is None and sigma_scale_factor > 0
    else:
        # chi-squared critical value for K degrees of freedom
        sigma_scale_factor = chi2.ppf(conf_interval, df=3) ** 0.5
    segment_thres = cal_prob_outside_conf_interval(
        3, sigma_scale_factor=sigma_scale_factor, conf_interval=None
    )

    for kpt in keypoints.tolist():
        if len(kpt) == 3:
            x, y, z = kpt
            kpt_cls = 0
            cov_mat = covariance
        elif len(kpt) == 4:
            x, y, z, kpt_cls = kpt
            cov_mat = covariance
        elif len(kpt) == 10:
            x, y, z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz, kpt_cls = kpt
            cov_mat = torch.tensor(
                [
                    [cov_xx, cov_xy, cov_xz],
                    [cov_xy, cov_yy, cov_yz],
                    [cov_xz, cov_yz, cov_zz],
                ],
                dtype=torch.float32,
            )
        else:
            raise ValueError

        if validate_cov_mat:
            _check_valid_covariance_matrix(cov_mat)

        kpt_cls = int(kpt_cls)
        assert kpt_cls < C

        if not same_std:
            # compute values interval along base axes XYZ, inside a confident inverval
            # e.g 99.9% <-> 4.0331422236561565 sigma-rule
            range_x, range_y, range_z = cal_range_in_conf_interval(
                cov_mat,
                conf_interval=None,
                sigma_scale_factor=sigma_scale_factor,
            ).tolist()
            assert range_x > 0 and range_y > 0 and range_z > 0
        else:
            principle_stds = cal_std_along_principle_axes(cov_mat)
            std = min(principle_stds)
            assert std > 0
            var = std**2
            cov_mat = torch.tensor(
                [
                    [var, 0, 0],
                    [0, var, 0],
                    [0, 0, var],
                ],
                dtype=torch.float32,
            )
            range_x = range_y = range_z = std * sigma_scale_factor

        # slices to crop -> reduce computation
        x_min = _clip(round(x - range_x), X)
        x_max = _clip(round(x + range_x), X, x_min + 1)
        y_min = _clip(round(y - range_y), Y)
        y_max = _clip(round(y + range_y), Y, y_min + 1)
        z_min = _clip(round(z - range_z), Z)
        z_max = _clip(round(z + range_z), Z, z_min + 1)
        if (x_max <= x_min) or (y_max <= y_min) or (z_max <= z_min):
            continue

        # Precompute covariance matrix inverse and determinant
        sigma_inv = torch.linalg.inv(cov_mat)
        grid_patch = grid[x_min:x_max, y_min:y_max, z_min:z_max]
        grid_diff = grid_patch - torch.tensor(
            [[[[x, y, z]]]], dtype=dtype
        )  # Shape (X, Y, Z, 3) - [1,1,1,3]

        # Compute multivariate Gaussian
        squared_mahalanobis = torch.einsum(
            "...i,ij,...j->...", grid_diff, sigma_inv, grid_diff
        )  # Shape (X, Y, Z)
        gaussian = torch.exp(-0.5 * squared_mahalanobis)  # min~0, max~1
        # convert gaussian heatmap to dense ellipsoid with 1 inside and 0 outside
        gaussian = (gaussian > segment_thres).to(gaussian.dtype)
        gaussian = lower + (upper - lower) * gaussian  # min=lower, max=upper
        heatmap_patch = heatmap[kpt_cls, x_min:x_max, y_min:y_max, z_min:z_max]
        torch.maximum(heatmap_patch, gaussian, out=heatmap_patch)

    return heatmap


def generate_3d_point_mask(
    heatmap_size: Tuple[int, int, int, int],
    keypoints: torch.Tensor,
    stride: int = 1,
    dtype=torch.float32,
    lower=0.0,
    upper=1.0,
    add_offset=True,
) -> torch.Tensor:
    """Generate 3D multivariate Gaussian heatmap from keypoints with arbitrary covariance matrix.

    Args:
        heatmap_size: heatmap size of shape (C, X, Y, Z) where C is the number of classes
        keypoints: (N, D) where D is one of:
            3 -> (x, y, z)
            4 -> (x, y, class)
            10 -> (x, y, z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz, class)
        dtype: output heatmap data type, use torch.float16 to save some memory else torch.float32
        lower: min heatmap value for label smoothing
        upper: max heatmap value for label smoothing
        add_offset: whether to add 0.5 offset to coordinates to transform from discreate pixel indices
            to floating point coordinate

    Returns:
        Tensor of shape (C, X, Y, Z) as provided by `heatmap_size`
    """
    assert 0 <= lower < upper <= 1.0
    assert len(keypoints.shape) == 2 and keypoints.shape[1] >= 3
    keypoints = keypoints.clone()
    # voxel indices -> floating point coordinate
    # origin is top-left corner of top-left voxel
    # center of first voxel (top-left) has coordinate (0.5, 0.5, 0.5)
    if add_offset:
        keypoints[:, :3] += 0.5
    keypoints[:, :3] /= torch.tensor([stride], dtype=torch.float32)
    C = heatmap_size[0]
    assert all([e % s == 0 for e, s in zip(heatmap_size[1:], stride)])
    X, Y, Z = [round(e / s) for e, s in zip(heatmap_size[1:], stride)]
    heatmap = torch.full((C, X, Y, Z), lower, dtype=dtype)
    if keypoints.shape[1] > 3:
        classes = keypoints[:, -1].long()
    else:
        classes = torch.zeros((keypoints.shape[1],), dtype=torch.long)
    indices = [classes, *torch.round(keypoints[:, :3].T - 0.5).long()]
    assert len(indices) == 4
    heatmap[indices] = upper
    return heatmap


def generate_2d_gaussian_heatmap(
    heatmap_size: Tuple[int, int, int, int],
    keypoints: torch.Tensor,
    stride: Tuple[int, int] = 1,
    dtype=torch.float32,
    sigma_scale_factor: float | None = None,
    conf_interval: float | None = 0.999,
    lower=0.0,
    upper=1.0,
    same_std=False,
    add_offset=True,
    conf_scale_mode=None,
    validate_cov_mat=False,
    output_tensor=None,
) -> torch.Tensor:
    """Generate 3D multivariate Gaussian heatmap from keypoints with arbitrary covariance matrix.

    Args:
        heatmap_size: heatmap size of shape (C, X, Y) where C is the number of classes
        keypoints: (N, 7) where 7 means
            (x, y, cov_xx, cov_yy, cov_xy, conf_scale, class)
        dtype: output heatmap data type, use torch.float16 to save some memory else torch.float32
        sigma_scale_factor: number of sigma-multiplicative rule, e.g 3 for dof=1, 4.03 for dof=3
        conf_interval: amount of confident interval drawed, covered by `sigma_scale_factor` sigma rule
        lower: min heatmap value for label smoothing
        upper: max heatmap value for label smoothing
        add_offset: whether to add 0.5 offset to coordinates to transform from discreate pixel indices
            to floating point coordinate
        validate_cov_mat: whether to validate the per-keypoint covariance matrix

    Returns:
        Tensor of shape (C, X, Y) as provided by `heatmap_size`
    """
    assert 0 <= lower < upper <= 1.0
    assert len(keypoints.shape) == 2 and keypoints.shape[1] == 7
    keypoints = keypoints.clone()
    # pixel indices -> floating point coordinate
    # origin is top-left corner of top-left pixel
    # center of first pixel (top-left) has coordinate (0.5, 0.5)
    if add_offset:
        keypoints[:, :2] += 0.5

    C = heatmap_size[0]
    assert all([e % s == 0 for e, s in zip(heatmap_size[1:], stride)])
    X, Y = [round(e / s) for e, s in zip(heatmap_size[1:], stride)]

    # Create 3D grid
    xs = torch.linspace(0.5, X - 0.5, X)
    ys = torch.linspace(0.5, Y - 0.5, Y)
    grid = torch.stack(
        torch.meshgrid([xs, ys], indexing="ij"), dim=-1
    )  # Shape (X, Y, 2)

    if output_tensor is None:
        heatmap = torch.full((C, X, Y), lower, dtype=dtype)
    else:
        heatmap = output_tensor
        assert tuple(heatmap.shape) == (C, X, Y) and heatmap.dtype == dtype
        heatmap[:] = lower

    keypoints[:, 0] /= stride[0]
    keypoints[:, 1] /= stride[1]
    keypoints[:, 2] /= stride[0] ** 2
    keypoints[:, 3] /= stride[1] ** 2
    keypoints[:, 4] /= stride[0] * stride[1]

    if sigma_scale_factor is not None:
        assert conf_interval is None and sigma_scale_factor > 0
    else:
        # chi-squared critical value for K degrees of freedom
        sigma_scale_factor = chi2.ppf(conf_interval, df=2) ** 0.5

    if conf_scale_mode == "segment":
        segment_thres = cal_prob_outside_conf_interval(
            3, sigma_scale_factor=sigma_scale_factor, conf_interval=None
        )

    for kpt in keypoints.tolist():
        x, y, cov_xx, cov_yy, cov_xy, conf_scale, kpt_cls = kpt
        cov_mat = torch.tensor(
            [
                [cov_xx, cov_xy],
                [cov_xy, cov_yy],
            ],
            dtype=torch.float32,
        )

        if validate_cov_mat:
            _check_valid_covariance_matrix(cov_mat)

        kpt_cls = int(kpt_cls)
        assert kpt_cls < C

        if not same_std:
            # compute values interval along base axes XYZ, inside a confident inverval
            # e.g 99.9% <-> 4.0331422236561565 sigma-rule
            range_x, range_y = cal_range_in_conf_interval(
                cov_mat,
                conf_interval=None,
                sigma_scale_factor=sigma_scale_factor,
            ).tolist()
            assert range_x > 0 and range_y > 0
        else:
            principle_stds = cal_std_along_principle_axes(cov_mat)
            std = min(principle_stds)
            assert std > 0
            var = std**2
            cov_mat = torch.tensor(
                [
                    [var, 0],
                    [0, var],
                ],
                dtype=torch.float32,
            )
            range_x = range_y = std * sigma_scale_factor

        # slices to crop -> reduce computation
        x_min = _clip(round(x - range_x), X)
        x_max = _clip(round(x + range_x), X, x_min + 1)
        y_min = _clip(round(y - range_y), Y)
        y_max = _clip(round(y + range_y), Y, y_min + 1)
        if (x_max <= x_min) or (y_max <= y_min):
            continue

        # Precompute covariance matrix inverse and determinant
        sigma_inv = torch.linalg.inv(cov_mat)
        grid_patch = grid[x_min:x_max, y_min:y_max]
        grid_diff = grid_patch - torch.tensor(
            [[[x, y]]], dtype=dtype
        )  # Shape (X, Y, 2) - [1,1,2]

        # Compute multivariate Gaussian
        squared_mahalanobis = torch.einsum(
            "...i,ij,...j->...", grid_diff, sigma_inv, grid_diff
        )  # Shape (X, Y)
        gaussian = torch.exp(-0.5 * squared_mahalanobis)  # min~0, max~1
        if conf_scale_mode is None or conf_scale_mode == "gaussian":
            gaussian *= conf_scale
        elif conf_scale_mode == "gaussian_min_max":
            # turn into [0,1] range
            _min = gaussian.min()
            _denominator = gaussian.max() - _min
            if _denominator > 0:
                gaussian = (gaussian - _min) / _denominator  # min=0, max=1
            else:
                # just do nothing, keep it as is
                logger.debug("Divide by zero!")
                pass
        elif conf_scale_mode == "segment":
            # convert gaussian heatmap to dense ellipsoid with 1 inside and 0 outside
            gaussian = (gaussian >= segment_thres).to(gaussian.dtype)
        else:
            raise ValueError
        gaussian = lower + (upper - lower) * gaussian  # min=lower, max=upper

        heatmap_patch = heatmap[kpt_cls, x_min:x_max, y_min:y_max]
        # print(gaussian.shape)
        torch.maximum(heatmap_patch, gaussian, out=heatmap_patch)

    return heatmap


def generate_2d_point_mask(
    heatmap_size: Tuple[int, int, int],
    keypoints: torch.Tensor,
    stride: Tuple[int, int] = (1, 1),
    dtype=torch.float32,
    lower=0.0,
    upper=1.0,
    add_offset=True,
    output_tensor=None,
) -> torch.Tensor:
    """Generate 2D multivariate point mask from keypoints with arbitrary covariance matrix.

    Args:
        heatmap_size: heatmap size of shape (C, X, Y) where C is the number of classes
        keypoints: (N, 7) where 7 means
            (x, y, cov_xx, cov_yy, cov_xy, conf_scale, class)
        dtype: output heatmap data type, use torch.float16 to save some memory else torch.float32
        lower: min heatmap value for label smoothing
        upper: max heatmap value for label smoothing
        add_offset: whether to add 0.5 offset to coordinates to transform from discreate pixel indices
            to floating point coordinate

    Returns:
        Tensor of shape (C, X, Y, Z) as provided by `heatmap_size`
    """
    assert 0 <= lower < upper <= 1.0
    assert len(keypoints.shape) == 2 and keypoints.shape[1] == 7
    keypoints = keypoints.clone()
    # voxel indices -> floating point coordinate
    # origin is top-left corner of top-left voxel
    # center of first voxel (top-left) has coordinate (0.5, 0.5, 0.5)
    if add_offset:
        keypoints[:, :2] += 0.5
    keypoints[:, :2] /= torch.tensor([stride], dtype=torch.float32)
    C = heatmap_size[0]
    assert all([e % s == 0 for e, s in zip(heatmap_size[1:], stride)])
    X, Y = [round(e / s) for e, s in zip(heatmap_size[1:], stride)]
    if output_tensor is None:
        heatmap = torch.full((C, X, Y), lower, dtype=dtype)
    else:
        heatmap = output_tensor
        assert tuple(heatmap.shape) == (C, X, Y) and heatmap.dtype == dtype
        heatmap[:] = lower
    classes = keypoints[:, -1].long()
    indices = torch.round(keypoints[:, :2].T - 0.5).long()
    torch.clip(indices[0, :], 0, Y, out=indices[0, :])
    torch.clip(indices[1, :], 0, X, out=indices[1, :])
    indices = [classes, *indices]
    assert len(indices) == 3
    heatmap[indices] = upper
    return heatmap


def generate_2d_heatmap(
    heatmap_size: Tuple[int, int, int, int],
    keypoints: torch.Tensor,
    stride: Tuple[int, int] = 1,
    dtype=torch.float32,
    sigma_scale_factor: float | None = None,
    conf_interval: float | None = 0.999,
    lower=0.0,
    upper=1.0,
    same_std=False,
    add_offset=True,
    conf_scale_mode=None,
    validate_cov_mat=False,
    output_tensor=None,
) -> torch.Tensor:
    """Generate 2D multivariate Gaussian heatmap from keypoints with arbitrary covariance matrix.

    Args:
        heatmap_size: heatmap size of shape (C, X, Y) where C is the number of classes
        keypoints: (N, 7) where 7 means
            (x, y, cov_xx, cov_yy, cov_xy, conf_scale, class)
        stride: heatmap stride, e.g 2,4,8,.. relative to `heatmap_size`
        dtype: output heatmap data type, use torch.float16 to save some memory else torch.float32
        sigma_scale_factor: number of sigma-multiplicative rule, e.g 3 for dof=1, 4.03 for dof=3
        conf_interval: amount of confident interval drawed, covered by `sigma_scale_factor` sigma rule
        lower: min heatmap value for label smoothing
        upper: max heatmap value for label smoothing
        add_offset: whether to add 0.5 offset to coordinates to transform from discreate pixel indices
            to floating point coordinate
        validate_cov_mat: whether to validate the per-keypoint covariance matrix

    Returns:
        Tensor of shape (C, X, Y) as provided by `heatmap_size`
    """
    if conf_scale_mode == "point":
        # efficient impl for point heatmap
        return generate_2d_point_mask(
            heatmap_size,
            keypoints,
            stride,
            dtype,
            lower,
            upper,
            add_offset,
            output_tensor,
        )
    return generate_2d_gaussian_heatmap(
        heatmap_size,
        keypoints,
        stride,
        dtype,
        sigma_scale_factor,
        conf_interval,
        lower,
        upper,
        same_std,
        add_offset,
        conf_scale_mode,
        validate_cov_mat,
        output_tensor,
    )


def z_slice_normal_dist_3d(mu_zyx, cov_zyx, z_slice):
    """
    Compute parameters of the 2D slice of a 3D normal distribution at Z = Vz,
    where input order is [Z, Y, X].

    Parameters
    ----------
    mu3 : array-like, shape (3,)
        Mean vector in [μz, μy, μx] order.

    Sigma3 : array-like, shape (3, 3)
        Covariance matrix in [Z, Y, X] variable order.

    Vz : float
        The fixed value for Z (slice at z = Vz).

    Returns
    -------
    C : float
        Scale factor = marginal PDF of Z at Z = Vz.

    mu2 : ndarray, shape (2,)
        Conditional mean vector [μy|Vz, μx|Vz].

    cov2 : ndarray, shape (2, 2)
        Conditional covariance matrix Σ_{YX|Z}.
    """
    mu_z = mu_zyx[0]
    mu_yx = mu_zyx[1:]  # [μy, μx]

    Sigma_zz = cov_zyx[0, 0]  # scalar
    Sigma_zyx = cov_zyx[0, 1:]  # 1×2 (Z vs [Y,X])
    Sigma_yx = cov_zyx[1:, 1:]  # 2×2 ([Y,X] block)

    # 1. Marginal density of Z at Vz
    diff_z = z_slice - mu_z
    # C = (1 / np.sqrt(2 * np.pi * Sigma_zz)) * np.exp(-0.5 * (diff_z**2) / Sigma_zz)
    C = np.exp(-0.5 * (diff_z**2) / Sigma_zz)

    # 2. Conditional mean
    mu2 = mu_yx + (Sigma_zyx / Sigma_zz) * diff_z

    # 3. Conditional covariance
    cov2 = Sigma_yx - np.outer(Sigma_zyx, Sigma_zyx) / Sigma_zz

    return C, mu2, cov2
