import math
from copy import deepcopy
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d

try:
    import tensorflow as tf
except:
    print("Could not import Tensorflow")

TF_INTERPOLATION_MODE_MAPPING = {
    "linear": "bilinear",
    "nearest": "nearest",
    "area": "area",
    "cubic": "bicubic",
}
TORCH_INTERPOLATION_MODE_MAPPING = {
    "linear": "linear",
    "nearest": "nearest",
    "nearest_exact": "nearest-exact",
    "area": "area",
}
CV2_INTERPOLATION_MODE_MAPPING = {
    "linear": cv2.INTER_LINEAR,
    "linear_exact": cv2.INTER_LINEAR_EXACT,
    "nearest": cv2.INTER_NEAREST,
    "nearest_exact": cv2.INTER_NEAREST_EXACT,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

SCIPY_INTERPOLATION_MODE_MAPPING = {
    "linear": "linear",
    "nearest": "nearest",
    "area": "area",
    "cubic": "cubic",
}

TF_INTERPOLATION_MODES = [
    "bilinear",
    "nearest",
    "bicubic",
    "area",
    "lanczos3",
    "lanczos5",
    "gaussian",
    "mitchellcubic",
]
TORCH_INTERPOLATION_MODES = [
    "nearest",
    "linear",
    "bilinear",
    "bicubic",
    "trilinear",
    "area",
    "nearest-exact",
]
SCIPY_INTERPOLATION_MODES = [
    "linear",
    "nearest",
    "nearest-up",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "previous",
    "next",
]
SCIPY_CONTAIN_NAN_INTERPOLATION_MODES = [
    "linear",
    "nearest",
    "nearest-up",
    "zero",
    "slinear",
    "previous",
    "next",
]
CV2_INTERPOLATION_MODES = list(CV2_INTERPOLATION_MODE_MAPPING.keys())
BACKEND_TO_INTERPOLATION_MODES = {
    "cv2": CV2_INTERPOLATION_MODES,
    "torch": TORCH_INTERPOLATION_MODES,
    "tf": TF_INTERPOLATION_MODES,
    "numpy": ["linear"],
    "scipy": SCIPY_INTERPOLATION_MODES,
}


def interp_1d(frames, target_len, mode="linear", backend="numpy"):
    """
    Interpolate the temporal dimension of shape (T, C) to (target_len, C)

    Parameters:
    - frames: np.ndarray of shape (T, C)
    - target_len: int, desired number of frames (T_out)
    - method: interpolation method (backend-specific)
    - backend: one of "cv2", "torch", "tf", "numpy", "scipy"

    Returns:
    - np.ndarray of shape (target_len, C)
    """
    # Input validation
    if not isinstance(frames, np.ndarray):
        raise TypeError("frames must be a numpy array")

    if frames.ndim != 2:
        raise ValueError(f"frames must be 2D array, got {frames.ndim}D")

    if not isinstance(target_len, int) or target_len < 1:
        raise ValueError("target_len must be a positive integer")

    if backend not in ["cv2", "torch", "tf", "numpy", "scipy"]:
        raise ValueError(
            f"backend must be one of ['cv2', 'torch', 'tf', 'numpy', 'scipy'], got '{backend}'"
        )

    T, C = frames.shape

    # Handle edge case: if already target length, return copy
    if T == target_len:
        return frames.copy()

    # Handle edge case: single frame input
    if T == 1:
        return np.tile(frames, (target_len, 1))

    if backend == "tf":
        mode = TF_INTERPOLATION_MODE_MAPPING.get(mode, mode)
        if mode not in TF_INTERPOLATION_MODES:
            raise ValueError(
                f"Unsupported tf method '{mode}'. Supported: {TF_INTERPOLATION_MODES}"
            )
        frames = frames.reshape(-1, 1, C)  # (T, 1, C)
        frames = tf.image.resize(frames, (target_len, 1), method=mode, antialias=False)
        frames = tf.squeeze(frames, axis=1).numpy()  # (target_len, C)

    elif backend == "torch":
        mode = TORCH_INTERPOLATION_MODE_MAPPING.get(mode, mode)
        if mode not in TORCH_INTERPOLATION_MODES:
            raise ValueError(
                f"Unsupported torch method '{mode}'. Supported: {TORCH_INTERPOLATION_MODES}"
            )
        frames_torch = torch.from_numpy(frames.astype(np.float32))
        frames_torch = frames_torch.transpose(0, 1).unsqueeze(0)  # (T, C) -> (1, C, T)
        interpolated = F.interpolate(
            frames_torch, size=target_len, mode=mode, align_corners=None
        )
        # Convert back to numpy
        frames = interpolated.squeeze(0).transpose(0, 1).numpy()  # (target_len, C)

    elif backend == "cv2":
        mode = CV2_INTERPOLATION_MODE_MAPPING.get(mode, mode)
        frames = np.expand_dims(frames, axis=1)  # (T, 1, C)
        frames = cv2.resize(frames, (1, target_len), interpolation=mode)
        frames = frames.squeeze(1)  # (target_len, C)

    elif backend == "numpy":
        # numpy.interp only supports linear interpolation and 1D arrays
        if mode != "linear":
            raise ValueError("numpy backend only supports 'linear' method")
        # Create coordinate arrays
        x_old = np.linspace(0, T - 1, T)
        x_new = np.linspace(0, T - 1, target_len)
        # Interpolate each channel separately
        frames_interp = np.zeros((target_len, C))
        for c in range(C):
            frames_interp[:, c] = np.interp(x_new, x_old, frames[:, c])
        frames = frames_interp

    elif backend == "scipy":
        mode = SCIPY_INTERPOLATION_MODE_MAPPING.get(mode, mode)
        if mode not in SCIPY_INTERPOLATION_MODES:
            raise ValueError(
                f"Unsupported scipy method '{mode}'. Supported: {SCIPY_INTERPOLATION_MODES}"
            )
        # Create coordinate arrays
        x_old = np.linspace(0, T - 1, T)
        x_new = np.linspace(0, T - 1, target_len)
        # Create interpolator for all channels at once
        interpolator = interp1d(
            x_old,
            frames,
            kind=mode,
            axis=0,
            copy=True,
            bounds_error=False,
            fill_value="extrapolate",
        )
        frames = interpolator(x_new)

    # Ensure output is the correct type and shape
    frames = np.asarray(frames, dtype=frames.dtype)
    if frames.shape != (target_len, C):
        raise RuntimeError(
            f"Output shape mismatch. Expected ({target_len}, {C}), got {frames.shape}"
        )
    return frames


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def butter_lowpass_filter(data, fmax=20, sr=200, order=4):
    nyquist = sr / 2
    normal_cutoff = fmax / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


if __name__ == "__main__":
    # Example usage and testing function
    def test_interp_1d():
        """Test the interpolation function with different backends and methods"""
        # Create test data
        T, C = 123, 13
        frames = np.random.randn(T, C).astype(np.float32)
        target_len = 234

        print(f"Original shape: {frames.shape}")
        print(f"Target length: {target_len}")

        # Test different backends
        backends_methods = {
            "numpy": ["linear"],
            "scipy": ["linear", "nearest", "cubic"],
            "cv2": ["linear", "nearest", "cubic"],
            # "torch": ["linear", "nearest"],  # Uncomment if PyTorch is available
            # "tf": ["bilinear", "nearest"],   # Uncomment if TensorFlow is available
        }

        for backend in ["torch", "cv2", "numpy", "scipy"]:
            print(f"\nTesting {backend} backend:")
            # for method in ['linear', 'nearest', 'area', 'cubic']:
            for method in list(
                set(
                    ["linear", "nearest", "area", "cubic"]
                    + TF_INTERPOLATION_MODES
                    + TORCH_INTERPOLATION_MODES
                    + SCIPY_INTERPOLATION_MODES
                )
            ):
                try:
                    result = interp_1d(frames, target_len, mode=method, backend=backend)
                    print(f"  {method}: {result.shape} ✓")
                except Exception as e:
                    e = str(e).split("\n")[0]
                    print(f"  {method}: Error - {e}")

    test_interp_1d()
