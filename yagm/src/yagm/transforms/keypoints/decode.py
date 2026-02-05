import logging
import time
from typing import Callable, List, Optional, Tuple

import torch
from torch.nn import functional as F
from yagm.transforms.keypoints.nms import keypoints_nms

try:
    import cc3d
except:
    print("Could not found cc3d")

logger = logging.getLogger(__name__)


@torch.inference_mode()
def decode_heatmap_3d_batched(
    heatmap: torch.Tensor,
    pool_ksize: Optional[Tuple[int, int, int]] = None,
    nms_radius_thres: Optional[float] = None,
    blur_operator: Optional[Callable] = None,
    conf_thres: float = 0.01,
    max_dets: Optional[int] = None,
    timeout: Optional[float] = None,
) -> List[List[torch.Tensor]]:
    """
    Decode a batched 3D probability heatmap into per-channel local maxima keypoints.

    This function processes a 5D heatmap tensor of shape `(B, C, D, H, W)` and returns,
    for each batch `b` and channel `c`, a list of detected 3D keypoints representing
    local maxima above a confidence threshold. Optional blur and local-max pooling
    are applied in a vectorized manner, while the actual decoding (top-1 or NMS)
    is intentionally performed per `(b, c)` using simple Python loops for clarity
    and debuggability.

    The output is always returned **on CPU**, regardless of whether the input
    heatmap is stored on CPU or CUDA.

    Parameters
    ----------
    heatmap : torch.Tensor
        Input tensor of shape `(B, C, D, H, W)` representing 3D probability maps.
        Channels are processed independently within each batch element.

    pool_ksize : tuple of int or None, optional
        A 3-tuple `(kz, ky, kx)` defining the odd-sized kernel for local-max pooling.
        If `None`, no pooling is performed. The pooling is channel-wise and
        preserves spatial resolution via stride=1 and symmetric padding.

    nms_radius_thres : float or None, optional
        Radius for radius-based non-maximum suppression. If `None` or `<= 0`,
        NMS is skipped. This is applied independently per `(b, c)`.

    blur_operator : callable or None, optional
        Optional function taking `(B, C, D, H, W)` → `(B, C, D, H, W)`. Intended for
        3D blurring, smoothing, or Gaussian filtering. Applied before peak detection.

    conf_thres : float, optional
        Minimum confidence required for a voxel to be considered a candidate peak.

    max_dets : int or None, optional
        Maximum number of detections to keep **per (b, c)**.
        - If `max_dets == 1`, a fast-path is used that returns the single strongest
          peak per channel (found via argmax), with no NMS.
        - Otherwise, NMS or top-K truncation is applied depending on
          `nms_radius_thres`.

    timeout : float or None, optional
        Optional timeout passed to `keypoints_nms`. If exceeded, NMS may early-exit.

    Returns
    -------
    List[List[torch.Tensor]]
        A nested list of shape `(B, C)`.
        Each entry `results[b][c]` is a CPU tensor of shape `(N, 4)` containing:

        ```
        [z, y, x, conf]
        ```

        where `(z, y, x)` are coordinates of a detected local maximum and `conf`
        is its heatmap value.

        If a channel has no valid detections, an empty tensor `(0, 4)` is returned.

    Notes
    -----
    - All heavy spatial operations (blur, max-pool) are applied vectorized.
    - All detection logic (argmax, candidate gathering, NMS) is performed per
      `(b, c)` using a simple, debuggable Python for-loop.
    - Output is always moved to CPU to avoid mixing device contexts in downstream
      post-processing.
    """
    if heatmap.ndim != 5:
        raise ValueError("heatmap must be (B, C, D, H, W)")

    B, C, D, H, W = heatmap.shape
    dtype = torch.float32

    # ---------------------------------------------------------
    # Output container: list[B][C] of tensors
    # ---------------------------------------------------------
    results: List[List[torch.Tensor]] = [
        [torch.empty((0, 4), dtype=dtype) for _ in range(C)] for _ in range(B)
    ]

    # ---------------------------------------------------------
    # 1) Vectorized blur
    # ---------------------------------------------------------
    if blur_operator is not None:
        heatmap = blur_operator(heatmap)

    # ---------------------------------------------------------
    # FAST PATH: per (b,c) top-1
    # ---------------------------------------------------------
    if max_dets == 1:
        for b in range(B):
            for c in range(C):
                hm = heatmap[b, c]  # (D,H,W)
                # Get flat argmax
                flat_idx = torch.argmax(hm)
                # Convert to (z, y, x)
                z, y, x = torch.unravel_index(flat_idx, hm.shape)
                # Get confidence score
                conf = hm[z, y, x]
                if conf < conf_thres:
                    continue
                # Store result (always CPU tensor)
                results[b][c] = torch.tensor(
                    [[z, y, x, conf]],
                    dtype=dtype,
                )
        return results

    # ---------------------------------------------------------
    # 2) Vectorized local max pooling → mask
    # ---------------------------------------------------------
    if pool_ksize is not None:
        if not all((k % 2 == 1) for k in pool_ksize):
            raise ValueError("pool_ksize must be odd integers")

        padding = [(k - 1) // 2 for k in pool_ksize]
        maximum = F.max_pool3d(
            heatmap, kernel_size=pool_ksize, stride=1, padding=padding
        )
        mask = (heatmap == maximum) & (heatmap >= conf_thres)
    else:
        mask = heatmap >= conf_thres

    # ---------------------------------------------------------
    # NMS PATH: iterate over (b,c) groups
    # ---------------------------------------------------------
    for b in range(B):
        for c in range(C):
            mask_bc = mask[b, c]  # (D,H,W)
            if not mask_bc.any():
                continue

            # Candidate coordinates
            nz = torch.nonzero(mask_bc, as_tuple=False)  # (Ncand, 3)
            if nz.numel() == 0:
                continue

            z = nz[:, 0].to(dtype)
            y = nz[:, 1].to(dtype)
            x = nz[:, 2].to(dtype)
            conf = heatmap[b, c, nz[:, 0], nz[:, 1], nz[:, 2]].to(dtype)

            keypoints = torch.stack([z, y, x, conf], dim=1)  # (N,4)

            # --- Radius NMS per channel ---
            if nms_radius_thres is not None and nms_radius_thres > 0:
                keep = keypoints_nms(
                    keypoints,
                    nms_radius_thres,
                    max_dets=max_dets,
                    timeout=timeout,
                )
                if len(keep) == 0:
                    continue
                kp = keypoints[keep]
            else:
                # no NMS, just optional top-K
                if max_dets is not None:
                    order = torch.argsort(keypoints[:, 3], descending=True)
                    order = order[:max_dets]
                    kp = keypoints[order]
                else:
                    kp = keypoints

            results[b][c] = kp.cpu()
    return results


def decode_segment_mask_3d(
    heatmap: torch.Tensor,
    conf_thres: float,
    volume_thres: float,
    max_dets: int = 1,
    conf_mode: str = "fuse",
    prob_mode: str = "avg",
    volume_upper_bound_factor=1.0,
):
    assert volume_upper_bound_factor >= 1.0
    assert conf_mode in ["volume", "prob", "fuse"]
    assert prob_mode in ["mean", "max", "center"]
    Z, Y, X = heatmap.shape

    # @TODO - carefully check the documents: https://github.com/seung-lab/connected-components-3d
    components = cc3d.connected_components(
        (heatmap >= conf_thres).cpu().numpy(), binary_image=True
    )
    stats = cc3d.statistics(components)
    num_components = len(stats["voxel_counts"])
    assert num_components >= 1
    if num_components == 1:
        # only background
        return torch.zeros((0, 4), dtype=torch.float32)

    voxel_counts = torch.tensor(stats["voxel_counts"], dtype=torch.float32)
    # the most common one should be background, on the first index
    assert voxel_counts.argmax() == 0

    # keep only large enough mass
    volumes = []
    component_idxs = []
    centroids = []
    for i in range(1, num_components):
        if voxel_counts[i] >= volume_thres:
            volumes.append(voxel_counts[i])
            component_idxs.append(i)
            centroids.append(stats["centroids"][i])
    volumes = torch.tensor(volumes, dtype=torch.float32)

    logger.debug(
        "\nSEGMENT DECODE: %d -> %d components, conf_mode=%s prob_mode=%s\n",
        num_components,
        len(volumes),
        conf_mode,
        prob_mode,
    )

    # calculate confident based on mass volume (voxel counts)
    volume_confs = volumes / (
        volume_thres * volume_upper_bound_factor
    )  # in range [1/volume_upper_bound_factor, inf]

    # calculate confidents based on probability map if needed
    if conf_mode in ["prob", "fuse"]:
        probs = []
        for component_idx, v, centroid in zip(component_idxs, volumes, centroids):
            if prob_mode == "center":
                z, y, x = centroid
                rz = min(max(0, round(z)), Z - 1)
                ry = min(max(0, round(y)), Y - 1)
                rx = min(max(0, round(x)), X - 1)
                prob = heatmap[rz, ry, rx]
            elif prob_mode == "mean":
                # prob = torch.sum(heatmap * (components_torch == component_idx)) / v
                prob = heatmap[components == component_idx].sum() / v
            elif prob_mode == "max":
                # prob = torch.max(heatmap * (components_torch == component_idx))
                prob = heatmap[components == component_idx].max()
            else:
                raise ValueError
            probs.append(prob.cpu())
        probs = torch.tensor(probs, dtype=torch.float32)
        assert len(probs) == len(volume_confs)

    if conf_mode == "volume":
        confs = volume_confs
    elif conf_mode == "prob":
        confs = probs
    elif conf_mode == "fuse":
        confs = (
            torch.minimum(volume_confs, torch.tensor(1.0, dtype=torch.float32)) * probs
        )
    else:
        raise ValueError

    if max_dets == 1:
        select_idx = confs.argmax()
        z, y, x = centroids[select_idx]
        conf = confs[select_idx]
        return torch.tensor([[z, y, x, conf]], dtype=torch.float32)
    else:
        raise NotImplementedError


@torch.inference_mode()
def decode_heatmap_2d_batched(
    heatmap: torch.Tensor,
    pool_ksize: Optional[Tuple[int, int]] = None,
    nms_radius_thres: Optional[float] = None,
    blur_operator: Optional[Callable] = None,
    conf_thres: float | None = 0.01,
    max_dets: Optional[int] = None,
    timeout: Optional[float] = None,
) -> List[List[torch.Tensor]]:
    """
    Decode a batched 2D probability heatmap into per-channel local maxima keypoints.

    This function processes a 4D heatmap tensor of shape `(B, C, H, W)` and returns,
    for each batch `b` and channel `c`, a list of detected 2D keypoints representing
    local maxima above a confidence threshold. Optional blur and local-max pooling
    are applied in a vectorized manner, while the actual decoding (top-1 or NMS)
    is intentionally performed per `(b, c)` using simple Python loops for clarity
    and debuggability.

    The output is always returned **on CPU**, regardless of whether the input
    heatmap is stored on CPU or CUDA.

    Parameters
    ----------
    heatmap : torch.Tensor
        Input tensor of shape `(B, C, H, W)` representing 2D probability maps.
        Channels are processed independently within each batch element.

    pool_ksize : tuple of int or None, optional
        A 2-tuple `(ky, kx)` defining the odd-sized kernel for local-max pooling.
        If `None`, no pooling is performed. The pooling is channel-wise and
        preserves spatial resolution via stride=1 and symmetric padding.

    nms_radius_thres : float or None, optional
        Radius for radius-based non-maximum suppression. If `None` or `<= 0`,
        NMS is skipped. This is applied independently per `(b, c)`.

    blur_operator : callable or None, optional
        Optional function taking `(B, C, H, W)` → `(B, C, H, W)`. Intended for
        2D blurring, smoothing, or Gaussian filtering. Applied before peak detection.

    conf_thres : float, optional
        Minimum confidence required for a pixel to be considered a candidate peak.

    max_dets : int or None, optional
        Maximum number of detections to keep **per (b, c)**.
        - If `max_dets == 1`, a fast-path is used that returns the single strongest
          peak per channel (found via argmax), with no NMS.
        - Otherwise, NMS or top-K truncation is applied depending on
          `nms_radius_thres`.

    timeout : float or None, optional
        Optional timeout passed to `keypoints_nms`. If exceeded, NMS may early-exit.

    Returns
    -------
    List[List[torch.Tensor]]
        A nested list of shape `(B, C)`.
        Each entry `results[b][c]` is a CPU tensor of shape `(N, 3)` containing:

        ```
        [y, x, conf]
        ```

        where `(y, x)` are coordinates of a detected local maximum and `conf`
        is its heatmap value.

        If a channel has no valid detections, an empty tensor `(0, 3)` is returned.

    Notes
    -----
    - All heavy spatial operations (blur, max-pool) are applied vectorized.
    - All detection logic (argmax, candidate gathering, NMS) is performed per
      `(b, c)` using a simple, debuggable Python for-loop.
    - Output is always moved to CPU to avoid mixing device contexts in downstream
      post-processing.
    """
    if heatmap.ndim != 4:
        raise ValueError("heatmap must be (B, C, H, W)")

    B, C, H, W = heatmap.shape
    dtype = torch.float32

    # ---------------------------------------------------------
    # Output container: list[B][C] of tensors
    # ---------------------------------------------------------
    results: List[List[torch.Tensor]] = [
        [torch.empty((0, 3), dtype=dtype) for _ in range(C)] for _ in range(B)
    ]

    # ---------------------------------------------------------
    # 1) Vectorized blur
    # ---------------------------------------------------------
    if blur_operator is not None:
        heatmap = blur_operator(heatmap)

    # ---------------------------------------------------------
    # FAST PATH: per (b,c) top-1
    # ---------------------------------------------------------
    if max_dets == 1:
        for b in range(B):
            for c in range(C):
                hm = heatmap[b, c]  # (H, W)
                flat_idx = torch.argmax(hm)
                y, x = torch.unravel_index(flat_idx, hm.shape)
                conf = hm[y, x]
                if conf_thres is not None and conf < conf_thres:
                    continue
                results[b][c] = torch.tensor(
                    [[y.item(), x.item(), conf.item()]],
                    dtype=dtype,
                )
        return results
    else:
        assert conf_thres is not None

    # ---------------------------------------------------------
    # 2) Vectorized local max pooling → mask
    # ---------------------------------------------------------
    if pool_ksize is not None:
        if not all((k % 2 == 1) for k in pool_ksize):
            raise ValueError("pool_ksize must be odd integers")

        padding = [(k - 1) // 2 for k in pool_ksize]
        # F.max_pool2d expects (N, C, H, W)
        maximum = F.max_pool2d(
            heatmap, kernel_size=pool_ksize, stride=1, padding=padding
        )
        mask = (heatmap == maximum) & (heatmap >= conf_thres)
    else:
        mask = heatmap >= conf_thres

    # ---------------------------------------------------------
    # NMS PATH: iterate over (b,c) groups
    # ---------------------------------------------------------
    for b in range(B):
        for c in range(C):
            mask_bc = mask[b, c]  # (H, W)
            if not mask_bc.any():
                continue

            # Candidate coordinates
            nz = torch.nonzero(mask_bc, as_tuple=False)  # (Ncand, 2) -> [y, x]
            if nz.numel() == 0:
                continue

            y = nz[:, 0].to(dtype)
            x = nz[:, 1].to(dtype)
            conf = heatmap[b, c, nz[:, 0], nz[:, 1]].to(dtype)

            keypoints = torch.stack([y, x, conf], dim=1)  # (N,3) -> [y, x, conf]

            # --- Radius NMS per channel ---
            if nms_radius_thres is not None and nms_radius_thres > 0:
                # keypoints_nms should accept (N,3) in the 2D case: [y, x, conf].
                keep = keypoints_nms(
                    keypoints,
                    nms_radius_thres,
                    max_dets=max_dets,
                    timeout=timeout,
                )
                if len(keep) == 0:
                    continue
                kp = keypoints[keep]
            else:
                # no NMS, just optional top-K
                if max_dets is not None:
                    order = torch.argsort(keypoints[:, 2], descending=True)
                    order = order[:max_dets]
                    kp = keypoints[order]
                else:
                    kp = keypoints

            results[b][c] = kp.cpu()

    return results
