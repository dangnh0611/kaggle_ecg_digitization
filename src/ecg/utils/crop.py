import logging
import os

import cv2
import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline, griddata

from ecg.utils import constants
from ecg.utils.interpolate import warp_2d

logger = logging.getLogger(__name__)


def _compute_crop_left_top(x_left, y_mid, signal_length_secs, crop_h, crop_w=512):
    # number of signal steps: secs * fs
    # step = 2200 / 11.176 * (1 / fs)
    # signal_pixels = step * secs * fs = 2200 / 11.176 * (1 / fs) * secs * fs
    signal_pixels = 2200 / 11.176 * signal_length_secs
    pad_left = (crop_w - signal_pixels) / 2
    crop_left = x_left - pad_left
    crop_top = y_mid - crop_h / 2
    return crop_left, crop_top


def crop_by_global_homo(
    lead_name,
    img,
    detected_keypoints,
    signal_length_secs=2.5,
    warp_h=512,
    warp_w=512,
    crop_h=512,
    crop_w=512,
    filter_nearby=True,
    interpolation_mode=cv2.INTER_CUBIC,
    nearby_keypoint_indices=constants.MANUAL_LEAD_TO_NEARBY_MAIN_KEYPOINT_INDICES,
):
    # reference keypoints in REFERENCE IMAGE 0001
    ref_kpt_xys = constants.REF_KPT_XYS
    # detected keypoints in current image
    detected_kpt_xys = detected_keypoints
    assert len(ref_kpt_xys) == len(detected_keypoints)
    if filter_nearby:
        nearby_kpt_indices = nearby_keypoint_indices[lead_name]
        ref_kpt_xys = ref_kpt_xys[nearby_kpt_indices]
        detected_kpt_xys = detected_kpt_xys[nearby_kpt_indices]

    ref_x_left, ref_y_mid = constants.LEAD_LEFT_MID_KEYPOINTS[lead_name]
    crop_left, crop_top = _compute_crop_left_top(
        ref_x_left, ref_y_mid, signal_length_secs, crop_h, crop_w
    )
    # print("CROP LTHW:", crop_left, crop_top, crop_h, crop_w)

    # now, left_top has contigous coordinate of [0, 0]
    inside_crop_ref_kpt_xys = ref_kpt_xys - np.array([[crop_left, crop_top]])
    # scale
    inside_warp_ref_kpt_xys = inside_crop_ref_kpt_xys * np.array(
        [[warp_w / crop_w, warp_h / crop_h]], dtype="float32"
    )

    # find homo transformation matrix from current image to warped image
    # -0.5 to convert from contigous physic coordiate system to pixel indices system
    for reproj_thres in [2, 4, 6, 8, 10]:
        H, mask = cv2.findHomography(
            detected_kpt_xys - 0.5,
            inside_warp_ref_kpt_xys - 0.5,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=reproj_thres,
            confidence=0.9999,
            maxIters=10_000,
        )
        if H is not None:
            # print(f"match: {mask.sum()} / {len(mask)}, H={H.shape}")
            break
    else:
        logger.info(
            "crop_by_global_homo() failed to find Homography with %d points, fallback to Perspective Transform.\nSRC=%s\nDST=%s\n",
            len(detected_kpt_xys),
            detected_kpt_xys,
            inside_warp_ref_kpt_xys,
        )
        H = cv2.getPerspectiveTransform(
            detected_kpt_xys[:4].astype("float32") - 0.5,
            inside_warp_ref_kpt_xys[:4].astype("float32") - 0.5,
        )

    warp_img = cv2.warpPerspective(
        img,
        H,
        (warp_w, warp_h),
        flags=interpolation_mode,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warp_img


def unwrap_with_griddata(
    img,
    detected_grid,
    reference_grid,
    warp_grid,
    griddata_method="linear",
):
    points = reference_grid.reshape(-1, 2)
    values_x = detected_grid[:, :, 0].flatten()
    values_y = detected_grid[:, :, 1].flatten()
    xi = warp_grid.reshape(-1, 2)

    map_x_flat = griddata(points, values_x, xi, method=griddata_method)
    map_y_flat = griddata(points, values_y, xi, method=griddata_method)

    # Handle NaNs
    if np.any(np.isnan(map_x_flat)) or np.any(np.isnan(map_y_flat)):
        print("Nan found during unwrap_with_griddata()")
        nan_mask = np.isnan(map_x_flat)
        map_x_flat[nan_mask] = griddata(
            points, values_x, xi[nan_mask], method="nearest"
        )
        map_y_flat[nan_mask] = griddata(
            points, values_y, xi[nan_mask], method="nearest"
        )

    out_h, out_w = warp_grid.shape[:2]
    map_x = map_x_flat.reshape(out_h, out_w).astype(np.float32)
    map_y = map_y_flat.reshape(out_h, out_w).astype(np.float32)
    return map_x, map_y


def unwrap_with_rect_spline(img, detected_grid, reference_grid, warp_grid):
    grid_rows = reference_grid[:, 0, 1]
    grid_cols = reference_grid[0, :, 0]

    if not (np.all(np.diff(grid_rows) > 0) and np.all(np.diff(grid_cols) > 0)):
        raise ValueError("Reference grid axes must be strictly increasing.")

    src_x_vals = detected_grid[:, :, 0]
    src_y_vals = detected_grid[:, :, 1]

    spline_x = RectBivariateSpline(grid_rows, grid_cols, src_x_vals, kx=3, ky=3, s=0)
    spline_y = RectBivariateSpline(grid_rows, grid_cols, src_y_vals, kx=3, ky=3, s=0)

    query_x = warp_grid[:, :, 0].flatten()
    query_y = warp_grid[:, :, 1].flatten()

    map_x_flat = spline_x(query_y, query_x, grid=False)
    map_y_flat = spline_y(query_y, query_x, grid=False)

    out_h, out_w = warp_grid.shape[:2]
    map_x = map_x_flat.reshape(out_h, out_w).astype(np.float32)
    map_y = map_y_flat.reshape(out_h, out_w).astype(np.float32)
    return map_x, map_y


def unwrap_with_piecewise_homography(
    img, detected_grid, reference_grid, warp_grid, K=4
):
    grid_rows = reference_grid[:, 0, 1]
    grid_cols = reference_grid[0, :, 0]
    rows = len(grid_rows)
    cols = len(grid_cols)

    H_matrices = np.zeros((rows - 1, cols - 1, 3, 3), dtype=np.float32)
    search_radius = int(np.sqrt(K)) + 2

    for r in range(rows - 1):
        for c in range(cols - 1):
            if K == 4:
                # Fast path code (same as previous)
                dst_pts = np.array(
                    [
                        detected_grid[r, c],
                        detected_grid[r, c + 1],
                        detected_grid[r + 1, c + 1],
                        detected_grid[r + 1, c],
                    ],
                    dtype=np.float32,
                )
                src_pts = np.array(
                    [
                        [grid_cols[c], grid_rows[r]],
                        [grid_cols[c + 1], grid_rows[r]],
                        [grid_cols[c + 1], grid_rows[r + 1]],
                        [grid_cols[c], grid_rows[r + 1]],
                    ],
                    dtype=np.float32,
                )
                # use solveMethod = cv2.DECOMP_LU by default
                H_matrices[r, c] = cv2.getPerspectiveTransform(src_pts, dst_pts)
            else:
                # --- ROBUST PATH: K Nearest Neighbors ---
                center_r, center_c = r + 0.5, c + 0.5
                r_min = max(0, int(center_r - search_radius))
                r_max = min(rows, int(center_r + search_radius + 1))
                c_min = max(0, int(center_c - search_radius))
                c_max = min(cols, int(center_c + search_radius + 1))

                rr, cc = np.meshgrid(
                    np.arange(r_min, r_max), np.arange(c_min, c_max), indexing="ij"
                )
                rr_flat, cc_flat = rr.flatten(), cc.flatten()

                dists_sq = (rr_flat - center_r) ** 2 + (cc_flat - center_c) ** 2
                k_indices = np.argsort(dists_sq)[:K]

                sel_r, sel_c = rr_flat[k_indices], cc_flat[k_indices]

                src_pts = reference_grid[sel_r, sel_c].astype(np.float32)
                dst_pts = detected_grid[sel_r, sel_c].astype(np.float32)

                H, _ = cv2.findHomography(
                    src_pts,
                    dst_pts,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=2,
                    confidence=0.9999,
                    maxIters=10_000,
                )
                if H is None:
                    # Fallback to K=4 logic
                    dst_bk = np.array(
                        [
                            detected_grid[r, c],
                            detected_grid[r, c + 1],
                            detected_grid[r + 1, c + 1],
                            detected_grid[r + 1, c],
                        ],
                        dtype=np.float32,
                    )
                    src_bk = np.array(
                        [
                            [grid_cols[c], grid_rows[r]],
                            [grid_cols[c + 1], grid_rows[r]],
                            [grid_cols[c + 1], grid_rows[r + 1]],
                            [grid_cols[c], grid_rows[r + 1]],
                        ],
                        dtype=np.float32,
                    )
                    H = cv2.getPerspectiveTransform(src_bk, dst_bk)

                H_matrices[r, c] = H

    x_coords = warp_grid[:, :, 0]
    y_coords = warp_grid[:, :, 1]

    col_indices = np.searchsorted(grid_cols, x_coords, side="right") - 1
    row_indices = np.searchsorted(grid_rows, y_coords, side="right") - 1
    col_indices = np.clip(col_indices, 0, cols - 2)
    row_indices = np.clip(row_indices, 0, rows - 2)
    # print(row_indices, col_indices, sep="\n")
    pixel_H = H_matrices[row_indices, col_indices]

    ones = np.ones_like(x_coords)
    points_homogeneous = np.stack([x_coords, y_coords, ones], axis=-1)[..., np.newaxis]
    mapped_homogeneous = np.matmul(pixel_H, points_homogeneous)

    w_vec = mapped_homogeneous[:, :, 2, 0]
    w_vec[w_vec == 0] = 1e-8

    map_x = (mapped_homogeneous[:, :, 0, 0] / w_vec).astype(np.float32)
    map_y = (mapped_homogeneous[:, :, 1, 0] / w_vec).astype(np.float32)
    return map_x, map_y


def crop_by_unwrap_flow(
    lead_name,
    img,
    detected_keypoints,
    signal_length_secs=2.5,
    warp_h=512,
    warp_w=512,
    crop_h=512,
    crop_w=512,
    filter_inside=True,
    unwrap_method="pw_homo",
    interpolation="lanczos",
    warp_backend="cv2",
    warp_device="cpu",
    flow_smooth_sigma=0.0,
    _shift_0dot5_in_cv2_remap=True,
    **unwrap_kwargs,
):

    ref_x_left, ref_y_mid = constants.LEAD_LEFT_MID_KEYPOINTS[lead_name]
    crop_left, crop_top = _compute_crop_left_top(
        ref_x_left, ref_y_mid, signal_length_secs, crop_h, crop_w
    )
    # print("CROP LTHW:", crop_left, crop_top, crop_h, crop_w)

    candidate_ref_kpt_xys = constants.REF_KPT_XYS[:2365]
    if filter_inside:
        is_kpt_inside_crop_indices = np.nonzero(
            np.all(
                (candidate_ref_kpt_xys >= np.array([[crop_left, crop_top]]))
                & (
                    candidate_ref_kpt_xys
                    <= np.array([[crop_left + crop_w, crop_top + crop_h]])
                ),
                axis=1,
            )
        )[0]
        # inside_crop_names = [REF_KPT_NAMES[idx] for idx in is_ref_kpt_inside_crop_indices]

        inside_kpt_idxs = constants.REF_FLAT_GRID_KPT_INDICES[
            is_kpt_inside_crop_indices
        ]
        inside_kpt_idx_min = inside_kpt_idxs.min()
        inside_kpt_idx_max = inside_kpt_idxs.max()
        # extend 1 cell along all 4 sides
        _row_start = max(0, inside_kpt_idx_min // 55 - 1)
        _row_end = min(inside_kpt_idx_max // 55 + 2, 43)
        _row_num = _row_end - _row_start
        _col_start = max(0, inside_kpt_idx_min % 55 - 1)
        _col_end = min(inside_kpt_idx_max % 55 + 2, 55)
        _col_num = _col_end - _col_start

        inside_kpt_idxs2 = constants.REF_GRID_KPT_INDICES[
            _row_start:_row_end, _col_start:_col_end
        ].flatten()
        inside_crop_ref_xys = candidate_ref_kpt_xys[inside_kpt_idxs2].reshape(
            _row_num, _col_num, 2
        )
        inside_detected_keypoints = detected_keypoints[inside_kpt_idxs2].reshape(
            _row_num, _col_num, 2
        )
    else:
        inside_crop_ref_xys = candidate_ref_kpt_xys.reshape(43, 55, 2)
        inside_detected_keypoints = detected_keypoints[:2365].reshape(43, 55, 2)

    # create the expected output reference grid points, in contigous format
    warp_ref_grid_x = np.linspace(crop_left + 0.5, crop_left + crop_w - 0.5, warp_w)
    warp_ref_grid_y = np.linspace(crop_top + 0.5, crop_top + crop_h - 0.5, warp_h)
    warp_ref_grid_xys = np.stack(
        np.meshgrid(warp_ref_grid_x, warp_ref_grid_y, indexing="xy"), axis=-1
    )

    # estimate unwrap flow (map)
    if unwrap_method == "pw_homo":
        unwrap_func = unwrap_with_piecewise_homography
    elif unwrap_method == "rect_spline":
        unwrap_func = unwrap_with_rect_spline
    elif unwrap_method == "griddata":
        unwrap_func = unwrap_with_griddata
    else:
        raise ValueError

    map_x, map_y = unwrap_func(
        img,
        inside_detected_keypoints,
        inside_crop_ref_xys,
        warp_ref_grid_xys,
        **unwrap_kwargs,
    )

    # Smoothing
    if flow_smooth_sigma > 0:
        ksize = max(3, int(2 * round(3 * flow_smooth_sigma) + 1))
        map_x = cv2.GaussianBlur(map_x, (ksize, ksize), flow_smooth_sigma)
        map_y = cv2.GaussianBlur(map_y, (ksize, ksize), flow_smooth_sigma)

    warp_img = warp_2d(
        img,
        map_x,
        map_y,
        interpolation=interpolation,
        backend=warp_backend,
        device=warp_device,
        _shift_0dot5_in_cv2_remap=_shift_0dot5_in_cv2_remap,
    )
    return warp_img
