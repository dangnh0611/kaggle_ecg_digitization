import math
import os

import matplotlib.pyplot as plt
import numpy as np


import math
import os
import matplotlib.pyplot as plt
import numpy as np

def show_image_grid(
    images,
    nrows=None,
    ncols=None,
    figsize_per_image=(2.5, 2.5),  # (width, height) hint; height is used, width is derived from aspect
    titles=None,
    save_path=None,
    convert_bgr=False,
    aspect_mode="median",           # {"median","mean","max","min"} to choose the common aspect ratio
):
    """
    Display images in an adaptive grid using matplotlib with a single, data-driven cell aspect ratio.

    Parameters
    ----------
    images : list[np.ndarray]
        List of images (HxW or HxWx3).
    nrows, ncols : int or None
        If None → compute automatically based on number of images.
        If one is provided, compute the other.
    figsize_per_image : tuple(float, float)
        (width, height) in inches per cell. Height is respected; width is recomputed from data aspect.
        Pass a larger height (figsize_per_image[1]) to make everything bigger.
    titles : list[str] or None
        Optional titles for each image.
    save_path : str or None
        If given, save the figure to this path instead of just showing it.
    convert_bgr : bool
        If True, convert BGR → RGB before showing. (OpenCV images)
    aspect_mode : str
        How to aggregate per-image aspect ratios (W/H) into a single common aspect:
        {"median","mean","max","min"}.
    """
    num_images = len(images)
    if num_images == 0:
        raise ValueError("No images to display.")

    # --- Auto compute rows and columns if not provided ---
    if nrows is None and ncols is None:
        ncols = math.ceil(math.sqrt(num_images))
        nrows = math.ceil(num_images / ncols)
    elif nrows is None:
        nrows = math.ceil(num_images / ncols)
    elif ncols is None:
        ncols = math.ceil(num_images / nrows)

    # --- Derive a single common aspect ratio (width/height) from the images ---
    aspects = []
    for im in images:
        if im.ndim == 2:
            h, w = im.shape
        elif im.ndim == 3:
            h, w = im.shape[:2]
        else:
            raise ValueError(f"Unsupported image ndim={im.ndim}")
        # Avoid zero division
        aspects.append((w if w > 0 else 1) / (h if h > 0 else 1))

    if aspect_mode == "mean":
        common_ar = float(np.mean(aspects))
    elif aspect_mode == "max":
        common_ar = float(np.max(aspects))
    elif aspect_mode == "min":
        common_ar = float(np.min(aspects))
    else:
        # default: median
        common_ar = float(np.median(aspects))

    # --- Figure size based on number of images and common aspect ---
    # Use the provided height; recompute width from aspect.
    per_cell_height = float(figsize_per_image[1])
    per_cell_width  = max(1e-6, common_ar * per_cell_height)  # guard tiny/zero

    figsize = (ncols * per_cell_width, nrows * per_cell_height)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # flatten safely even if nrows/ncols == 1

    for i, ax in enumerate(axes):
        if i < num_images:
            img = images[i]

            # Convert BGR → RGB if requested
            if convert_bgr and img.ndim == 3 and img.shape[2] == 3:
                img = img[:, :, ::-1]

            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)

            ax.axis("off")

            if titles is not None and i < len(titles):
                ax.set_title(titles[i], fontsize=8)
        else:
            ax.axis("off")

    plt.tight_layout()

    # --- Save or show the figure ---
    if save_path is not None:
        dir_ = os.path.dirname(save_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
    else:
        plt.show()

