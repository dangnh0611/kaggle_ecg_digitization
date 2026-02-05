import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import zoom  # for interpolation


def viz_img_heatmap(
    rgb,
    heatmap,
    save_path=None,
    overlay_alpha=0.5,
    figsize=(15, 15),  # size per subplot
    only_overlay=False,
):
    """
    Visualize an RGB image and a multi-class heatmap.

    If only_overlay is False:
        1x4 grid:
            [ RGB
              Heatmap (channels 1..C-1 blended)
              Heatmap (channel 0, grayscale)
              RGB + all-channel heatmap overlay ]

    If only_overlay is True:
        1x1 figure:
            [ RGB + all-channel heatmap overlay ]

    Parameters
    ----------
    rgb : np.ndarray or torch.Tensor
        RGB image, shape (H, W, 3) or (3, H, W).

    heatmap : np.ndarray or torch.Tensor
        Heatmap tensor of shape (C, H, W).

    save_path : str or None
        If provided, saves the visualization as an image.

    overlay_alpha : float
        Alpha for blending heatmap (all channels) onto RGB in the overlay view.

    figsize : tuple(float, float)
        Size of a single subplot in inches (width, height).
        Total figure size is computed from this and the layout.

    only_overlay : bool
        If True, display only the overlayed image (no grid).

    Returns
    -------
    fig, axes
        Matplotlib figure and axes. For only_overlay=True, axes is a single Axes.
        For only_overlay=False, axes is a 1x4 array of Axes.
    """
    # Convert to numpy
    rgb = np.asarray(rgb)
    heatmap = np.asarray(heatmap)

    # Force float32 for scipy.ndimage.zoom compatibility
    if heatmap.dtype not in (np.float32, np.float64):
        heatmap = heatmap.astype(np.float32, copy=False)

    # ---- Normalize / reshape RGB ------------------------------------------------
    if rgb.ndim != 3:
        raise ValueError(f"rgb must be 3D, got {rgb.shape}")

    # Allow either (3, H, W) or (H, W, 3)
    if rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))

    H, W, _ = rgb.shape

    # Normalize RGB to [0,1]
    if rgb.dtype == np.uint8:
        rgb_vis = rgb.astype(np.float32) / 255.0
    else:
        rgb_vis = np.clip(rgb.astype(np.float32), 0.0, 1.0)

    # ---- Check / resize heatmap ------------------------------------------------
    if heatmap.ndim != 3:
        raise ValueError(f"heatmap must be 3D (C, H, W), got {heatmap.shape}")

    C, H_hm, W_hm = heatmap.shape

    if (H_hm, W_hm) != (H, W):
        # Resize heatmap to match RGB size
        scale_y = H / float(H_hm)
        scale_x = W / float(W_hm)
        heatmap = zoom(heatmap, (1.0, scale_y, scale_x), order=1)
        C, H_hm, W_hm = heatmap.shape

    # ---- Base colors for all channels ------------------------------------------
    base_colors = [
        (0.0, 1.0, 0.0),  # green
        (1.0, 0.0, 0.0),  # red
        (0.0, 0.0, 1.0),  # blue
        (1.0, 1.0, 0.0),  # yellow
        (1.0, 0.0, 1.0),  # magenta
        (0.0, 1.0, 1.0),  # cyan
        (1.0, 0.5, 0.0),  # orange
        (0.5, 0.0, 1.0),  # purple
    ]

    # ---- Composite heatmap for channels 1..C-1 ---------------------------------
    composite_rest = np.zeros((H, W, 3), dtype=np.float32)
    if C > 1:
        for c in range(1, C):
            hm = heatmap[c]
            hm_min, hm_max = float(hm.min()), float(hm.max())

            if hm_max > hm_min:
                hm_norm = (hm - hm_min) / (hm_max - hm_min)
            else:
                hm_norm = np.zeros_like(hm)

            color = base_colors[c % len(base_colors)]

            class_rgb = np.stack(
                [
                    hm_norm * color[0],
                    hm_norm * color[1],
                    hm_norm * color[2],
                ],
                axis=-1,
            )

            composite_rest += class_rgb

        composite_rest = np.clip(composite_rest, 0.0, 1.0)

    # ---- Full composite heatmap for all channels (0..C-1) ----------------------
    composite_all = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(C):
        hm = heatmap[c]
        hm_min, hm_max = float(hm.min()), float(hm.max())

        if hm_max > hm_min:
            hm_norm = (hm - hm_min) / (hm_max - hm_min)
        else:
            hm_norm = np.zeros_like(hm)

        color = base_colors[c % len(base_colors)]

        class_rgb = np.stack(
            [
                hm_norm * color[0],
                hm_norm * color[1],
                hm_norm * color[2],
            ],
            axis=-1,
        )

        composite_all += class_rgb

    composite_all = np.clip(composite_all, 0.0, 1.0)

    # ---- Overlay: RGB + all-channel heatmap ------------------------------------
    overlay = (1.0 - overlay_alpha) * rgb_vis + overlay_alpha * composite_all
    overlay = np.clip(overlay, 0.0, 1.0)

    # ---- Heatmap[0] grayscale normalizing to [0, 255] --------------------------
    hm0 = heatmap[0]
    hm0_min, hm0_max = float(hm0.min()), float(hm0.max())

    if hm0_max > hm0_min:
        hm0_norm = (hm0 - hm0_min) / (hm0_max - hm0_min)
    else:
        hm0_norm = np.zeros_like(hm0)

    hm0_gray = (hm0_norm * 255.0).astype(np.uint8)

    # ---- Plot -------------------------------------------------------------------
    if only_overlay:
        # Single subplot, use "figsize" directly
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(overlay)
        ax.set_title(f"RGB + all-channel heatmap (alpha={overlay_alpha})")
        ax.axis("off")
        axes = ax
    else:
        # 1x4 grid; total fig size is 4 * per-subplot width, 1 * per-subplot height
        total_figsize = (figsize[0] * 4, figsize[1])
        fig, axes = plt.subplots(1, 4, figsize=total_figsize)

        # (0): RGB
        axes[0].imshow(rgb_vis)
        axes[0].set_title("RGB")
        axes[0].axis("off")

        # (1): Heatmap channels 1..C-1 blended
        axes[1].imshow(composite_rest)
        axes[1].set_title("Heatmap (channels 1..C-1)")
        axes[1].axis("off")

        # (2): Heatmap[0] grayscale
        axes[2].imshow(hm0_gray, cmap="gray")
        axes[2].set_title("Heatmap (channel 0, grayscale)")
        axes[2].axis("off")

        # (3): RGB + all-channel heatmap overlay
        axes[3].imshow(overlay)
        axes[3].set_title(f"RGB + all-channel heatmap (alpha={overlay_alpha})")
        axes[3].axis("off")

    plt.tight_layout()

    # ---- Save if requested ------------------------------------------------------
    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()
    return fig, axes


def viz_img_keypoints(
    img,
    kpt_xys,
    kpt_classes,
    save_path=None,
    figsize=(8, 8),
    point_size=30,
    alpha=0.9,
    show_legend=False,
):
    """
    Visualize N keypoints on an RGB image using matplotlib.

    Args:
        img (np.ndarray): RGB image, shape (H, W, 3)
        kpt_xys (array-like): (N, 2) keypoints in (x, y) pixel coordinates
        kpt_classes (array-like): (N,) integer class labels
        save_path (str | None): path to save visualization
        figsize (tuple): matplotlib figure size
        point_size (int): scatter point size
        alpha (float): point transparency

    Returns:
        fig, axes
    """
    img = np.asarray(img)
    kpt_xys = np.asarray(kpt_xys)
    kpt_classes = np.asarray(kpt_classes)

    assert img.ndim == 3 and img.shape[2] == 3, "img must be HxWx3 RGB"
    assert kpt_xys.shape[0] == kpt_classes.shape[0]
    assert kpt_xys.shape[1] == 2

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    axes.imshow(img)
    axes.axis("off")

    # Unique classes → consistent colors
    unique_classes = np.unique(kpt_classes)
    cmap = plt.get_cmap("tab10")

    class_to_color = {cls: cmap(i % cmap.N) for i, cls in enumerate(unique_classes)}

    # Plot per class for clean legend & coloring
    for cls in unique_classes:
        mask = kpt_classes == cls
        if not np.any(mask):
            continue

        pts = kpt_xys[mask]
        axes.scatter(
            pts[:, 0],
            pts[:, 1],
            s=point_size,
            c=[class_to_color[cls]],
            alpha=alpha,
            label=f"class {cls}",
        )

    if show_legend:
        axes.legend(loc="upper right", fontsize=8, frameon=False)

    # ---- Save if requested ------------------------------------------------------
    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()
    return fig, axes


def viz_pred_ecg_img_keypoints(
    img,
    grid_points,
    main_points_dict,
    save_path=None,
    figsize=(8, 8),
    point_size=30,
    alpha=0.9,
    show_legend=False,
):
    """
    Visualize keypoints in a 2x2 grid.

    Layout:
        (1) Original image + grid_points
        (2) Original image + main_points (method 1)
        (3) Original image + main_points (method 2)
        (4) Original image + main_points (method 3)

    Args:
        img (np.ndarray): RGB image, shape (H, W, 3)
        grid_points (array-like): (Ng, 2) grid keypoints
        main_points_dict (dict[str, array-like]):
            Mapping method_name -> (N, 2) keypoints.
            At most 3 methods. All must have same N.
    """
    img = np.asarray(img)
    grid_points = np.asarray(grid_points)

    assert img.ndim == 3 and img.shape[2] == 3, "img must be HxWx3 RGB"
    assert grid_points.ndim == 2 and grid_points.shape[1] == 2

    assert len(main_points_dict) <= 3, "main_points_dict can have at most 3 methods"

    method_names = list(main_points_dict.keys())
    main_points = [np.asarray(main_points_dict[k]) for k in method_names]

    # Validate main points
    if main_points:
        n_points = main_points[0].shape[0]
        for pts in main_points:
            assert pts.shape == (
                n_points,
                2,
            ), "All main_points must have same shape (N,2)"

    # ------------------------------------------------------------
    # Color setup: SAME color index = SAME keypoint index
    # ------------------------------------------------------------
    if main_points:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(n_points)]
    else:
        colors = []

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # ------------------------------------------------------------
    # (1) Original + grid points
    # ------------------------------------------------------------
    ax = axes[0]
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Grid points")

    if len(grid_points) > 0:
        ax.scatter(
            grid_points[:, 0],
            grid_points[:, 1],
            s=point_size,
            c="green",
            alpha=alpha,
        )

    # ------------------------------------------------------------
    # (2)(3)(4) Main points per method
    # ------------------------------------------------------------
    for i, (method, pts) in enumerate(zip(method_names, main_points), start=1):
        ax = axes[i]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(method)

        for j in range(pts.shape[0]):
            ax.scatter(
                pts[j, 0],
                pts[j, 1],
                s=point_size,
                color=colors[j],
                alpha=alpha,
                label=f"pt {j}" if show_legend else None,
            )

        if show_legend:
            ax.legend(
                loc="upper right",
                fontsize=6,
                frameon=False,
                ncol=2,
            )

    # Hide unused subplots (if < 3 methods)
    for k in range(1 + len(main_points), 4):
        axes[k].axis("off")

    plt.tight_layout()

    # ---- Save if requested ------------------------------------------------------
    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()
    return fig, axes


def viz_signal_heatmap(img, heatmap, layout="1x3"):
    """
    Displays a grid of [Base Image, Colored Heatmap, Overlay].

    Args:
        base_img: Numpy array (H, W) or (H, W, 3)
        heatmap: Numpy array or Torch tensor (H, W)
        layout: '1x3' (horizontal) or '3x1' (vertical)
    """
    # 1. Prepare Heatmap (Tensor -> Numpy -> Norm -> Color)
    if isinstance(heatmap, torch.Tensor):
        hm_data = heatmap.detach().cpu().numpy()
    else:
        hm_data = heatmap

    # Normalize to 0-255
    hm_norm = (hm_data - hm_data.min()) / (hm_data.max() - hm_data.min() + 1e-8)
    hm_uint8 = (hm_norm * 255).astype(np.uint8)

    # Apply Jet Colormap (Blue=Low, Red=High) & Convert to RGB
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    # 2. Prepare Base Image (Ensure RGB)
    img_rgb = img.copy()
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    # 3. Resize Heatmap to match Base Image (for overlay/grid)
    target_h, target_w = img_rgb.shape[:2]
    if hm_color.shape[:2] != (target_h, target_w):
        hm_color = cv2.resize(
            hm_color, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

    # 4. Create Overlay
    alpha, beta = 0.6, 0.4
    overlay = cv2.addWeighted(img_rgb, alpha, hm_color, beta, 0)

    # 5. Arrange Grid
    # Add simple borders/labels logic if needed, but here we just concat
    images = [img_rgb, hm_color, overlay]

    if layout == "1x3":
        grid = np.hstack(images)
    elif layout == "3x1":
        grid = np.vstack(images)
    else:
        raise ValueError("Layout must be '1x3' or '3x1'")

    return grid
