from typing import Any, Tuple, List
from itertools import product
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _validate_sliding_patch_positions(img_size: Tuple, patches: np.ndarray) -> None:
    """
    Validate if sliding returned patches are valid or not.
    For debuging purpose only.
    Args:
        img_size: the image size (Z, Y, X)
        patches: a numpy array of shape (N, 4) where N is the number of patches,
            4 means (patch_start, patch_end, crop_start, crop_end)

    Raise:
        Exception to indicate invalid patches.
    """
    img = np.zeros(img_size, dtype=np.uint16)
    for roi_start, roi_end, patch_start, patch_end in patches:
        if (patch_start > roi_start).any() or (patch_end < roi_end).any():
            raise Exception
        roi_start = [max(0, e) for e in roi_start]
        slices = tuple(slice(start, end) for start, end in zip(roi_start, roi_end))
        img[slices] += 1
    if np.any(img < 1):
        raise Exception


def get_sliding_patch_positions(
    img_size: Tuple[int],
    patch_size: Tuple[int],
    border: Tuple[int | float] = (0, 0, 0),
    overlap: Tuple[int | float] = (0, 0, 0),
    start: Tuple[int] = (0, 0, 0),
    validate=False,
) -> np.ndarray:
    """
    Get sliding window patches's position over an image.
    Usually used for sliding window inference process.

    Args:
        img_size: the image size, tested on 3D (Z, Y, X)
        patch_size: the cropped patch size, e.g, will be passed to model
        border: border usually used to provide more context for better prediction,
            but usually not contribute the the final aggregated prediction or loss,
            e.g when ambiguity happend in near edge/border regions.
        overlap: patch overlaping to mitigate bordering artifact or a kind of ensemble.
        start: starting offset position
        validate: whether to validate the returned patches, for debuging purpose only.
            Should be set to False (default).

    Returns:
        A numpy array of shape (N, 4) where N is the number of patches, and
            4 indicate [patch_start, patch_end, crop_start, crop_end].
            `CROP=PATCH + BORDER` and CROP can contain out-of-image coordinate.
            all patches should roll over all image pixels.
    """
    if any([o < b for o, b in zip(overlap, border)]):
        logger.warning("Overlap is smaller than border size: %s < %s", overlap, border)
    # ensure patchSize and startPos are the right length
    assert len(img_size) == len(patch_size) == len(border) == len(overlap) == len(start)

    # calculate boder value
    if isinstance(border[0], float):
        assert all([isinstance(e, float) for e in border])
        border = [round(ps * b) for ps, b in zip(patch_size, border)]
    border = np.array(border)
    patch_size = np.array(patch_size)
    roi_size = patch_size - 2 * border

    # calculate step value, which depends on the amount of overlap
    if isinstance(overlap[0], float):
        assert all([isinstance(e, float) for e in overlap])
        overlap = [round(ps * o) for ps, o in zip(patch_size, overlap)]
    step = tuple(round(ps - b - o) for ps, b, o in zip(patch_size, border, overlap))
    
    end = img_size
    ranges = []
    for _start, _end, _step, _roi_size in zip(start, end, step, roi_size):
        _range = list(range(_start, _end - _roi_size + _step, _step))
        # last patch will be ajusted, as much context (ROI region) as posible
        _range[-1] = _end - _roi_size
        ranges.append(_range)
    roi_starts = np.array(list(product(*ranges)))
    roi_ends = roi_starts + roi_size[None]
    patch_starts = roi_starts - border[None]
    patch_ends = roi_ends + border[None]
    # (N, 3) -> (N, 4, 3)
    ret = np.stack([roi_starts, roi_ends, patch_starts, patch_ends], axis=1)
    if validate:
        _validate_sliding_patch_positions(img_size, ret)
    return ret


# def get_sliding_patch_positions(
#     img_size: Tuple[int],
#     patch_size: Tuple[int],
#     border: Tuple[int | float] = (0, 0, 0),
#     overlap: Tuple[int | float] = (0, 0, 0),
#     start: Tuple[int] = (0, 0, 0),
#     validate=False,
#     adaptive = False,
#     stride = (1,1,1),
# ) -> np.ndarray:
#     """
#     Get sliding window patches's position over an image.
#     Usually used for sliding window inference process.

#     Args:
#         img_size: the image size, tested on 3D (Z, Y, X)
#         patch_size: the cropped patch size, e.g, will be passed to model
#         border: border usually used to provide more context for better prediction,
#             but usually not contribute the the final aggregated prediction or loss,
#             e.g when ambiguity happend in near edge/border regions.
#         overlap: patch overlaping to mitigate bordering artifact or a kind of ensemble.
#         start: starting offset position
#         validate: whether to validate the returned patches, for debuging purpose only.
#             Should be set to False (default).

#     Returns:
#         A numpy array of shape (N, 4) where N is the number of patches, and
#             4 indicate [patch_start, patch_end, crop_start, crop_end].
#             `CROP=PATCH + BORDER` and CROP can contain out-of-image coordinate.
#             all patches should roll over all image pixels.
#     """
#     if any([o < b for o, b in zip(overlap, border)]):
#         logger.warning("Overlap is smaller than border size: %s < %s", overlap, border)
#     # ensure patchSize and startPos are the right length
#     assert len(img_size) == len(patch_size) == len(border) == len(overlap) == len(start)

#     # calculate step value, which depends on the amount of overlap
#     if isinstance(overlap[0], float):
#         assert all([isinstance(e, float) for e in overlap])
#         step = tuple(round(p * (1.0 - 2 * o)) for p, o in zip(patch_size, overlap))
#     else:
#         step = tuple(p - 2 * o for p, o in zip(patch_size, overlap))

#     # calculate boder value
#     if isinstance(border[0], float):
#         assert all([isinstance(e, float) for e in border])
#         border = [round(p * b) for p, b in zip(patch_size, border)]
#     border = np.array(border)
#     patch_size = np.array(patch_size)
#     roi_size = patch_size - 2 * border

#     end = img_size
#     ranges = []
#     for _start, _end, _step, _roi_size in zip(start, end, step, roi_size):
#         _range = list(range(_start, _end - _roi_size + _step, _step))
#         # last patch will be ajusted, as much context (ROI region) as posible
#         if not adaptive:
#             _range[-1] = _end - _roi_size
#         else:
#             if len(_range) >= 2:
#                 prev_end = _range[-2] + _roi_size
#             else:
#                 prev_end = 0
#             last_roi_size = 10
            
#         ranges.append(_range)
#     roi_starts = np.array(list(product(*ranges)))
#     roi_ends = roi_starts + roi_size[None]
#     patch_starts = roi_starts - border[None]
#     patch_ends = roi_ends + border[None]
#     # (N, 3) -> (N, 4, 3)
#     ret = np.stack([roi_starts, roi_ends, patch_starts, patch_ends], axis=1)
#     if validate:
#         _validate_sliding_patch_positions(img_size, ret)
#     return ret