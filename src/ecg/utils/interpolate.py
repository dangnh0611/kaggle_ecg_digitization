import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

# Extended Interpolation Map
# Maps string names to appropriate backend constants
INTERPOLATION_MODES = {
    "nearest": {"cv2": cv2.INTER_NEAREST, "scipy": 0, "torch": "nearest"},
    "linear": {"cv2": cv2.INTER_LINEAR, "scipy": 1, "torch": "bilinear"},
    "bilinear": {"cv2": cv2.INTER_LINEAR, "scipy": 1, "torch": "bilinear"},
    "area": {
        "cv2": cv2.INTER_AREA,
        "scipy": 1,
        "torch": "bilinear",
    },  # Scipy/Torch don't strictly have area, linear is closest proxy
    "cubic": {"cv2": cv2.INTER_CUBIC, "scipy": 3, "torch": "bicubic"},
    "bicubic": {"cv2": cv2.INTER_CUBIC, "scipy": 3, "torch": "bicubic"},
    "lanczos": {
        "cv2": cv2.INTER_LANCZOS4,
        "scipy": 3,
        "torch": "bicubic",
    },  # Scipy order 3 is approx similar to Lanczos quality
    # Higher Order Splines (Mainly for SciPy)
    # CV2/Torch fall back to their highest quality available
    "spline4": {"cv2": cv2.INTER_LANCZOS4, "scipy": 4, "torch": "bicubic"},
    "spline5": {"cv2": cv2.INTER_LANCZOS4, "scipy": 5, "torch": "bicubic"},
}


def interp_1d_cv2(arr, target_len, method="adaptive"):
    """
    Resizes a 1D array.

    - Upsampling: Uses Cubic interpolation (smoothness).
    - Downsampling: Uses Area interpolation (preserves sum/intensity).
    """
    # Ensure input is a numpy array
    arr = np.array(arr)
    input_len = len(arr)

    # Bypass if no change
    if input_len == target_len:
        return arr

    # Reshape to (1, W) because cv2.resize expects an image (2D)
    # We treat the 1D array as a 1-pixel tall image.
    img_1d = arr.reshape(1, -1)

    # Choose interpolation method based on resizing direction
    if target_len < input_len:
        # DOWN-sampling: Use INTER_AREA.
        # This averages pixel values, ensuring peaks aren't "skipped".
        method = cv2.INTER_AREA
    else:
        # UP-sampling: Use INTER_LINEAR (or INTER_CUBIC).
        # This creates smooth transitions between points.
        method = cv2.INTER_CUBIC

    # Resize
    # Note: cv2.resize size argument is (Width, Height) -> (target_len, 1)
    result = cv2.resize(img_1d, (target_len, 1), interpolation=method)

    # Flatten back to 1D
    return result.flatten()


def _warp_2d_legacy(img, map_x, map_y, backend="cv2", device="cpu"):
    """
    Applies the remapping (warping) using the specified backend.

    Args:
        img: (H, W, C) or (H, W) numpy array.
        map_x: (H_out, W_out) float32 array of absolute X pixel coordinates.
        map_y: (H_out, W_out) float32 array of absolute Y pixel coordinates.
        backend: 'cv2', 'scipy', or 'torch'.
        device: 'cpu' or 'cuda' (only used if backend='torch').
    """
    # Ensure maps are float32
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # ---------------------------------------------------------
    # 1. OpenCV Backend (Fastest CPU)
    # ---------------------------------------------------------
    if backend == "cv2":
        # -0.5 to convert from contigous coordinate space to pixel indices
        return cv2.remap(
            img, map_x - 0.5, map_y - 0.5, interpolation=cv2.INTER_LANCZOS4
        )

    # ---------------------------------------------------------
    # 2. SciPy Backend (Precise, Scientific)
    # ---------------------------------------------------------
    elif backend == "scipy":
        # scipy.ndimage.map_coordinates expects coordinates in (Row, Col) order, i.e., (y, x).
        # It also expects a shape of (2, N_points), in contigous coordinate space
        # so we don't need  -0.5, keep contigous as is
        coords = np.stack([map_y, map_x])

        # Handle 2D (Grayscale) vs 3D (Color)
        if img.ndim == 2:
            # order=3 is Bicubic, mode='nearest' matches cv2 border logic
            return map_coordinates(img, coords, order=3, mode="nearest")
        elif img.ndim == 3:
            # Loop over channels (R, G, B)
            channels = []
            for c in range(img.shape[2]):
                channels.append(
                    map_coordinates(img[..., c], coords, order=3, mode="nearest")
                )
            return np.stack(channels, axis=-1)

    # ---------------------------------------------------------
    # 3. PyTorch Backend (GPU Accelerated, Differentiable)
    # ---------------------------------------------------------
    elif backend == "torch":
        H, W = img.shape[:2]

        # Convert Image to Tensor: (H, W, C) -> (1, C, H, W)
        if img.ndim == 2:
            img_tensor = torch.from_numpy(img)[None, None, ...].float()
        else:
            # Transpose to Channel-First
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))[None, ...].float()

        img_tensor = img_tensor.to(device)

        # Normalize Coordinates to [-1, 1] for grid_sample
        # We use align_corners=False convention
        norm_x = 2.0 * map_x / W - 1.0
        norm_y = 2.0 * map_y / H - 1.0

        # Stack to (1, H_out, W_out, 2)
        grid = np.stack([norm_x, norm_y], axis=-1)
        grid_tensor = torch.from_numpy(grid)[None, ...].float().to(device)

        # Sampling
        out_tensor = F.grid_sample(
            img_tensor,
            grid_tensor,
            mode="bicubic",
            padding_mode="border",
            align_corners=False,
        )

        # Convert back to Numpy: (1, C, H, W) -> (H, W, C)
        out_np = out_tensor.detach().cpu().numpy()[0]
        if img.ndim == 3:
            out_np = out_np.transpose(1, 2, 0)
        else:
            out_np = out_np[0]  # Squeeze channel dim for grayscale

        return np.clip(out_np, 0, 255).astype(img.dtype)

    else:
        raise ValueError(f"Unknown backend: {backend}")


def warp_2d(
    img,
    map_x,
    map_y,
    interpolation="lanczos",
    backend="cv2",
    device="cpu",
    _shift_0dot5_in_cv2_remap=False,
):
    """
    Applies the remapping (warping) using the specified backend.

    Args:
        img: (H, W, C) or (H, W) numpy array.
        map_x: (H_out, W_out) float32 array of absolute X pixel coordinates.
        map_y: (H_out, W_out) float32 array of absolute Y pixel coordinates.
        interpolation: String key from INTERPOLATION_MODES (e.g., 'linear', 'cubic', 'spline4', 'spline5').
        backend: 'cv2', 'scipy', or 'torch'.
        device: 'cpu' or 'cuda' (only used if backend='torch').
    """
    # Ensure maps are float32
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # Validate interpolation choice
    if interpolation not in INTERPOLATION_MODES:
        raise ValueError(
            f"Unknown interpolation method: {interpolation}. "
            f"Available: {list(INTERPOLATION_MODES.keys())}"
        )

    # Get backend-specific mode settings
    modes = INTERPOLATION_MODES[interpolation]

    # ---------------------------------------------------------
    # 1. OpenCV Backend (Fastest CPU)
    # ---------------------------------------------------------
    if backend == "cv2":
        cv2_interp = modes["cv2"]
        if _shift_0dot5_in_cv2_remap:
            # -0.5 to convert from contiguous coordinate space to pixel indices
            return cv2.remap(img, map_x - 0.5, map_y - 0.5, interpolation=cv2_interp)
        else:
            return cv2.remap(img, map_x, map_y, interpolation=cv2_interp)

    # ---------------------------------------------------------
    # 2. SciPy Backend (Precise, Supports Higher Order Splines)
    # ---------------------------------------------------------
    elif backend == "scipy":
        scipy_order = modes["scipy"]

        # scipy.ndimage.map_coordinates expects coordinates in (Row, Col) order -> (y, x).
        coords = np.stack([map_y, map_x])

        # Handle 2D (Grayscale) vs 3D (Color)
        if img.ndim == 2:
            return map_coordinates(img, coords, order=scipy_order, mode="nearest")
        elif img.ndim == 3:
            channels = []
            for c in range(img.shape[2]):
                channels.append(
                    map_coordinates(
                        img[..., c], coords, order=scipy_order, mode="nearest"
                    )
                )
            return np.stack(channels, axis=-1)

    # ---------------------------------------------------------
    # 3. PyTorch Backend (GPU Accelerated, Differentiable)
    # ---------------------------------------------------------
    elif backend == "torch":
        torch_mode = modes["torch"]
        H, W = img.shape[:2]

        # Convert Image to Tensor
        if img.ndim == 2:
            img_tensor = torch.from_numpy(img)[None, None, ...].float()
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))[None, ...].float()

        img_tensor = img_tensor.to(device)

        # Normalize Coordinates to [-1, 1]
        norm_x = 2.0 * map_x / W - 1.0
        norm_y = 2.0 * map_y / H - 1.0

        grid = np.stack([norm_x, norm_y], axis=-1)
        grid_tensor = torch.from_numpy(grid)[None, ...].float().to(device)

        # Sampling
        out_tensor = F.grid_sample(
            img_tensor,
            grid_tensor,
            mode=torch_mode,
            padding_mode="border",
            align_corners=False,
        )

        # Convert back to Numpy
        out_np = out_tensor.detach().cpu().numpy()[0]
        if img.ndim == 3:
            out_np = out_np.transpose(1, 2, 0)
        else:
            out_np = out_np[0]

        return np.clip(out_np, 0, 255).astype(img.dtype)

    else:
        raise ValueError(f"Unknown backend: {backend}")
