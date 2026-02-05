import math

import cv2
import numpy as np


def get_scale_xy_from_homo_mat(H):
    """
    Compute scale factors along X and Y for an affine-like homography H.
    No np.linalg calls, only simple arithmetic.
    """
    H = np.asarray(H, dtype=float)
    h00, h01 = H[0, 0], H[0, 1]
    h10, h11 = H[1, 0], H[1, 1]

    # scale_x = length of column 0
    scale_x = math.sqrt(h00 * h00 + h10 * h10)

    # scale_y = length of column 1
    scale_y = math.sqrt(h01 * h01 + h11 * h11)

    return scale_x, scale_y


def angular_diff(a, b):
    """Signed smallest difference a-b in degrees, in (-180, 180]."""
    return ((a - b + 180.0) % 360.0) - 180.0


def angle_to_rotation_idx(angle):
    return ((round(angle) + 45) % 360) // 90


def get_rotation_angle_from_homo_mat(H):
    """
    Returns the angle (degrees) of the source image relative to the reference.
    Positive = Clockwise rotation of Source.
    """
    # math.atan2(y, x) returns angle in radians
    # H[1,0] = sin component, H[0,0] = cos component
    transform_angle_rad = math.atan2(H[1, 0], H[0, 0])
    transform_angle_deg = math.degrees(transform_angle_rad)

    # Invert sign because H is the transform Source -> Reference
    return -transform_angle_deg


def rotate_image(image, angle, border_mode=cv2.BORDER_CONSTANT, border_val=0):
    """
    Rotate an image by 'angle' degrees (float).
    Keeps the full image without cropping.
    Args:
        image:
        angle: Positive values indicate counter-clockwise rotation, while negative values indicate clockwise rotation.
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # Rotation matrix (scale=1.0 keeps original size)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding box to avoid cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_val,
    )

    return rotated
