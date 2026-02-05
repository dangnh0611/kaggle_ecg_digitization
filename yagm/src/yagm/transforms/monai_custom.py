import logging
import math
import time
from abc import abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from math import ceil, sqrt
from typing import Any, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
from monai.config import DtypeLike, KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import (
    get_random_patch,
    get_valid_patch_size,
    orientation_ras_lps,
    to_affine_nd,
)
from monai.networks.layers.convutils import gaussian_1d
from monai.networks.layers.simplelayers import separable_filtering
from monai.transforms import Cropd, RandCropd, Spacing
from monai.transforms.inverse import InvertibleTransform, TraceableTransform
from monai.transforms.spatial.functional import rotate90
from monai.transforms.traits import LazyTrait, MultiSampleTrait
from monai.transforms.transform import (
    LazyTransform,
    MapTransform,
    Randomizable,
    RandomizableTrait,
    RandomizableTransform,
    Transform,
)
from monai.transforms.utils import (
    apply_affine_to_points,
    correct_crop_centers,
    create_translate,
    map_spatial_axes,
)
from monai.transforms.utils_pytorch_numpy_unification import (
    concatenate,
    in1d,
    linalg_inv,
    moveaxis,
    unravel_index,
    unravel_indices,
)
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    TraceKeys,
    TransformBackends,
    convert_data_type,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    issequenceiterable,
)
from monai.utils.enums import TraceKeys, TransformBackends
from monai.utils.type_conversion import (
    convert_to_dst_type,
    convert_to_tensor,
    get_dtype_string,
)
from torch import nn
from yagm.transforms.keypoints.encode import cal_range_in_conf_interval

logger = logging.getLogger(__name__)


class CustomGaussianFilter(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        sigma: Sequence[float] | float | Sequence[torch.Tensor] | torch.Tensor,
        truncated: float = 4.0,
        approx: str = "erf",
        requires_grad: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
                must have shape (Batch, channels, H[, W, ...]).
            sigma: std. could be a single value, or `spatial_dims` number of values.
            truncated: spreads how many stds.
            approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".

                - ``erf`` approximation interpolates the error function;
                - ``sampled`` uses a sampled Gaussian kernel;
                - ``scalespace`` corresponds to
                  https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel
                  based on the modified Bessel functions.

            requires_grad: whether to store the gradients for sigma.
                if True, `sigma` will be the initial value of the parameters of this module
                (for example `parameters()` iterator could be used to get the parameters);
                otherwise this module will fix the kernels using `sigma` as the std.
        """
        if issequenceiterable(sigma):
            if len(sigma) != spatial_dims:  # type: ignore
                raise ValueError
        else:
            sigma = [deepcopy(sigma) for _ in range(spatial_dims)]  # type: ignore
        super().__init__()
        self.sigma = [
            torch.nn.Parameter(
                torch.as_tensor(
                    s,
                    dtype=torch.float,
                    device=s.device if isinstance(s, torch.Tensor) else None,
                ),
                requires_grad=requires_grad,
            )
            for s in sigma  # type: ignore
        ]
        self.truncated = truncated
        self.approx = approx
        for idx, param in enumerate(self.sigma):
            self.register_parameter(f"kernel_sigma_{idx}", param)

        self._kernel = [
            gaussian_1d(s, truncated=self.truncated, approx=self.approx)
            for s in self.sigma
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape [Batch, chns, H, W, D].
        """
        return separable_filtering(x=x, kernels=self._kernel)


class CustomProbNMS(Transform):
    """
    Performs probability based non-maximum suppression (NMS) on the probabilities map via
    iteratively selecting the coordinate with highest probability and then move it as well
    as its surrounding values. The remove range is determined by the parameter `box_size`.
    If multiple coordinates have the same highest probability, only one of them will be
    selected.

    Args:
        spatial_dims: number of spatial dimensions of the input probabilities map.
            Defaults to 2.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, spatial_dims: int = 2) -> None:
        self.filters = {}
        self.spatial_dims = spatial_dims

    def __call__(
        self,
        prob_map: NdarrayOrTensor,
        prob_threshold=0.5,
        sigma: Sequence[float] | float | Sequence[torch.Tensor] | torch.Tensor = 0.0,
        box_size: int | Sequence[int] = 48,
        max_dets: int | None = None,
        timeout: float | None = None,
    ):
        """
        Args:
            prob_map: the input probabilities map, it must have shape (H[, W, ...]).
            prob_threshold: the probability threshold, the function will stop searching if
                the highest probability is no larger than the threshold. The value should be
                no less than 0.0. Defaults to 0.5.
            sigma: the standard deviation for gaussian filter.
                It could be a single value, or `spatial_dims` number of values. Defaults to 0.0.
            box_size: the box size (in pixel) to be removed around the pixel with the maximum probability.
                It can be an integer that defines the size of a square or cube,
                or a list containing different values for each dimensions. Defaults to 48.

        Return:
            a list of selected lists, where inner lists contain probability and coordinates.
            For example, for 3D input, the inner lists are in the form of [probability, x, y, z].

        """
        if sigma != 0:
            if not isinstance(prob_map, torch.Tensor):
                prob_map = torch.as_tensor(prob_map, dtype=torch.float)
            if sigma not in self.filters:
                self.filters[sigma] = CustomGaussianFilter(
                    spatial_dims=self.spatial_dims, sigma=sigma
                )
            filter = self.filters[sigma]
            filter.to(prob_map.device)
            prob_map = filter(prob_map)

        if isinstance(box_size, int):
            box_size = np.asarray([box_size] * self.spatial_dims)
        else:
            box_size = np.asarray(box_size)
        box_lower_bd = box_size // 2
        box_upper_bd = box_size - box_lower_bd

        prob_map_shape = prob_map.shape
        outputs = []
        start = time.time()
        while prob_map.max() > prob_threshold:
            if max_dets is not None and len(outputs) > max_dets:
                logger.warning("NMS: max_dets exceeded %d > %d", len(outputs), max_dets)
                break
            max_idx = unravel_index(prob_map.argmax(), prob_map_shape)
            prob_max = prob_map[tuple(max_idx)]
            max_idx = (
                max_idx.cpu().numpy() if isinstance(max_idx, torch.Tensor) else max_idx
            )
            prob_max = (
                prob_max.item() if isinstance(prob_max, torch.Tensor) else prob_max
            )
            outputs.append(list(max_idx) + [prob_max])

            idx_min_range = (max_idx - box_lower_bd).clip(0, None)
            idx_max_range = (max_idx + box_upper_bd).clip(None, prob_map_shape)
            # for each dimension, set values during index ranges to 0
            slices = tuple(
                slice(idx_min_range[i], idx_max_range[i])
                for i in range(self.spatial_dims)
            )
            prob_map[slices] = 0

            if timeout:
                time_diff = time.time() - start
                if time_diff > timeout:
                    logger.warning(
                        "NMS timeout exceeded %f > %f, num_outputs=%d",
                        time_diff,
                        timeout,
                        len(outputs),
                    )
                    break
        return outputs


class Rotate180(InvertibleTransform, LazyTransform):
    """
    Rotate an array by 180 degrees in the plane specified by `axes`.
    See `torch.rot90` for additional details:
    https://pytorch.org/docs/stable/generated/torch.rot90.html#torch-rot90.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self, k: int = 1, spatial_axes: tuple[int, int] = (0, 1), lazy: bool = False
    ) -> None:
        """
        Args:
            k: number of times to rotate by 180 degrees.
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
                If axis is negative it counts from the last to the first axis.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
        """
        LazyTransform.__init__(self, lazy=lazy)
        self.k = k % 2  # 0, 1
        spatial_axes_: tuple[int, int] = ensure_tuple(spatial_axes)
        if len(spatial_axes_) != 2:
            raise ValueError(
                f"spatial_axes must be 2 numbers to define the plane to rotate, got {spatial_axes_}."
            )
        self.spatial_axes = spatial_axes_

    def __call__(self, img: torch.Tensor, lazy: bool | None = None) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        axes = map_spatial_axes(img.ndim, self.spatial_axes)
        lazy_ = self.lazy if lazy is None else lazy
        # rotate180 = 2 * rotate90
        return rotate90(img, axes, self.k * 2, lazy=lazy_, transform_info=self.get_transform_info())  # type: ignore

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        return self.inverse_transform(data, transform)

    def inverse_transform(self, data: torch.Tensor, transform) -> torch.Tensor:
        axes = transform[TraceKeys.EXTRA_INFO]["axes"]
        k = transform[TraceKeys.EXTRA_INFO]["k"]
        inv_k = k % 2  # equivalent to 2 - k % 2
        xform = Rotate180(k=inv_k, spatial_axes=axes)
        with xform.trace_transform(False):
            return xform(data)


class RandRotate180(RandomizableTransform, InvertibleTransform, LazyTransform):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = Rotate180.backend

    def __init__(
        self,
        prob: float = 0.1,
        spatial_axes: tuple[int, int] = (0, 1),
        lazy: bool = False,
    ) -> None:
        """
        Args:
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`, (Default 3).
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
        """
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.spatial_axes = spatial_axes

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None

    def __call__(
        self, img: torch.Tensor, randomize: bool = True, lazy: bool | None = None
    ) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            randomize: whether to execute `randomize()` function first, default to True.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """

        if randomize:
            self.randomize()
        lazy_ = self.lazy if lazy is None else lazy
        if self._do_transform:
            xform = Rotate180(1, self.spatial_axes, lazy=lazy_)
            out = xform(img)
        else:
            out = convert_to_tensor(img, track_meta=get_track_meta())
        self.push_transform(out, replace=True, lazy=lazy_)
        return out

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        xform_info = self.pop_transform(data)
        if not xform_info[TraceKeys.DO_TRANSFORM]:
            return data
        rotate_xform = xform_info[TraceKeys.EXTRA_INFO]
        return Rotate180().inverse_transform(data, rotate_xform)


class RandRotate180d(
    RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform
):
    """
    Dictionary-based version :py:class:`monai.transforms.RandRotate90`.
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = Rotate180.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        spatial_axes: tuple[int, int] = (0, 1),
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)

        self.spatial_axes = spatial_axes

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None
    ) -> Mapping[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        self.randomize()
        d = dict(data)

        # FIXME: here we didn't use array version `RandRotate90` transform as others, because we need
        # to be compatible with the random status of some previous integration tests
        lazy_ = self.lazy if lazy is None else lazy
        rotator = Rotate180(1, self.spatial_axes, lazy=lazy_)
        for key in self.key_iterator(d):
            d[key] = (
                rotator(d[key])
                if self._do_transform
                else convert_to_tensor(d[key], track_meta=get_track_meta())
            )
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d

    def inverse(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if not isinstance(d[key], MetaTensor):
                continue
            xform = self.pop_transform(d[key])
            if xform[TraceKeys.DO_TRANSFORM]:
                d[key] = Rotate180().inverse_transform(
                    d[key], xform[TraceKeys.EXTRA_INFO]
                )
        return d


# CREDIT: https://github.com/matthew-brett/transforms3d/blob/main/transforms3d/affines.py
def decompose44(A44):
    """Decompose 4x4 homogenous affine matrix into parts.

    The parts are translations, rotations, zooms, shears.

    This is the same as :func:`decompose` but specialized for 4x4 affines.

    Decomposes `A44` into ``T, R, Z, S``, such that::

       Smat = np.array([[1, S[0], S[1]],
                        [0,    1, S[2]],
                        [0,    0,    1]])
       RZS = np.dot(R, np.dot(np.diag(Z), Smat))
       A44 = np.eye(4)
       A44[:3,:3] = RZS
       A44[:-1,-1] = T

    The order of transformations is therefore shears, followed by
    zooms, followed by rotations, followed by translations.

    This routine only works for shape (4,4) matrices

    Parameters
    ----------
    A44 : array shape (4,4)

    Returns
    -------
    T : array, shape (3,)
       Translation vector
    R : array shape (3,3)
        rotation matrix
    Z : array, shape (3,)
       Zoom vector.  May have one negative zoom to prevent need for negative
       determinant R matrix above
    S : array, shape (3,)
       Shear vector, such that shears fill upper triangle above
       diagonal to form shear matrix (type ``striu``).

    Examples
    --------
    >>> T = [20, 30, 40] # translations
    >>> R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] # rotation matrix
    >>> Z = [2.0, 3.0, 4.0] # zooms
    >>> S = [0.2, 0.1, 0.3] # shears
    >>> # Now we make an affine matrix
    >>> A = np.eye(4)
    >>> Smat = np.array([[1, S[0], S[1]],
    ...                  [0,    1, S[2]],
    ...                  [0,    0,    1]])
    >>> RZS = np.dot(R, np.dot(np.diag(Z), Smat))
    >>> A[:3,:3] = RZS
    >>> A[:-1,-1] = T # set translations
    >>> Tdash, Rdash, Zdash, Sdash = decompose44(A)
    >>> np.allclose(T, Tdash)
    True
    >>> np.allclose(R, Rdash)
    True
    >>> np.allclose(Z, Zdash)
    True
    >>> np.allclose(S, Sdash)
    True

    Notes
    -----
    The implementation inspired by:

    *Decomposing a matrix into simple transformations* by Spencer
    W. Thomas, pp 320-323 in *Graphics Gems II*, James Arvo (editor),
    Academic Press, 1991, ISBN: 0120644819.

    The upper left 3x3 of the affine consists of a matrix we'll call
    RZS::

       RZS = R * Z *S

    where R is a rotation matrix, Z is a diagonal matrix of scalings::

       Z = diag([sx, sy, sz])

    and S is a shear matrix of form::

       S = [[1, sxy, sxz],
            [0,   1, syz],
            [0,   0,   1]])

    Running all this through sympy (see 'derivations' folder) gives
    ``RZS`` as ::

       [R00*sx, R01*sy + R00*sx*sxy, R02*sz + R00*sx*sxz + R01*sy*syz]
       [R10*sx, R11*sy + R10*sx*sxy, R12*sz + R10*sx*sxz + R11*sy*syz]
       [R20*sx, R21*sy + R20*sx*sxy, R22*sz + R20*sx*sxz + R21*sy*syz]

    ``R`` is defined as being a rotation matrix, so the dot products between
    the columns of ``R`` are zero, and the norm of each column is 1.  Thus
    the dot product::

       R[:,0].T * RZS[:,1]

    that results in::

       [R00*R01*sy + R10*R11*sy + R20*R21*sy + sx*sxy*R00**2 + sx*sxy*R10**2 + sx*sxy*R20**2]

    simplifies to ``sy*0 + sx*sxy*1`` == ``sx*sxy``.  Therefore::

       R[:,1] * sy = RZS[:,1] - R[:,0] * (R[:,0].T * RZS[:,1])

    allowing us to get ``sy`` with the norm, and sxy with ``R[:,0].T *
    RZS[:,1] / sx``.

    Similarly ``R[:,0].T * RZS[:,2]`` simplifies to ``sx*sxz``, and
    ``R[:,1].T * RZS[:,2]`` to ``sy*syz`` giving us the remaining
    unknowns.
    """
    A44 = np.asarray(A44)
    T = A44[:-1, -1]
    RZS = A44[:-1, :-1]
    # compute scales and shears
    M0, M1, M2 = np.array(RZS).T
    # extract x scale and normalize
    sx = math.sqrt(np.sum(M0**2))
    M0 /= sx
    # orthogonalize M1 with respect to M0
    sx_sxy = np.dot(M0, M1)
    M1 -= sx_sxy * M0
    # extract y scale and normalize
    sy = math.sqrt(np.sum(M1**2))
    M1 /= sy
    sxy = sx_sxy / sx
    # orthogonalize M2 with respect to M0 and M1
    sx_sxz = np.dot(M0, M2)
    sy_syz = np.dot(M1, M2)
    M2 -= sx_sxz * M0 + sy_syz * M1
    # extract z scale and normalize
    sz = math.sqrt(np.sum(M2**2))
    M2 /= sz
    sxz = sx_sxz / sx
    syz = sy_syz / sy
    # Reconstruct rotation matrix, ensure positive determinant
    Rmat = np.array([M0, M1, M2]).T
    if np.linalg.det(Rmat) < 0:
        sx *= -1
        Rmat[:, 0] *= -1
    return T, Rmat, np.array([sx, sy, sz]), np.array([sxy, sxz, syz])


def custom_apply_affine_to_keypoints_3d(
    data: torch.Tensor, affine: torch.Tensor, dtype: torch.dtype = torch.float64
):
    """
    Apply affine transformation to a set of 3D keypoints with scaling.

    Args:
        data: Input data to apply affine transformation, should be a tensor of shape (C, N, K),
              where N is the number of points, last dim has K >= 6 channels corresponding to:
              - First 3 channels: X, Y, Z coordinates
              - Next 3 channels: X, Y, Z scale factors for each keypoint
        affine: Affine matrix to be applied, should be a tensor of shape (4, 4).
        dtype: Output data dtype (default: torch.float64).

    Returns:
        Tensor of transformed keypoints with the same shape as input.
    """
    assert len(data.shape) == 3
    data_: torch.Tensor = convert_to_tensor(data, track_meta=False, dtype=torch.float64)
    affine = to_affine_nd(3, affine)

    # Extract the 3D coordinates (first 3 channels)
    coordinates = data_[:, :, :3]

    # Convert data to homogeneous coordinates (add the extra 1s for affine transformation)
    homogeneous_coordinates = torch.cat(
        (
            coordinates,
            torch.ones(
                (coordinates.shape[0], coordinates.shape[1], 1), dtype=coordinates.dtype
            ),
        ),
        dim=2,
    )

    # Apply the affine transformation (matrix multiplication with affine transpose)
    transformed_homogeneous = torch.matmul(homogeneous_coordinates, affine.T)

    # Extract the transformed coordinates (remove the homogeneous coordinate)
    transformed_coordinates = transformed_homogeneous[:, :, :-1]

    transformed_data = [transformed_coordinates]
    # Optionally, apply scaling from affine matrix (e.g., if affine includes scaling components)
    if data_.shape[2] >= 6:
        scales = data_[:, :, 3:6]
        T, R, S, SHEAR = decompose44(affine)
        # print('SCALE HWD:', S)
        scaling_factors = torch.from_numpy(S.reshape(1, 1, 3))
        # Apply scaling to the keypoint scales if desired (scales can be adjusted based on affine scaling)
        transformed_scales = scales * scaling_factors
        transformed_data.append(transformed_scales)
    if data_.shape[2] > 6:
        transformed_data.append(data_[:, :, 6:])
    # Concatenate the transformed coordinates with unchanged class index and transformed scales
    transformed_data = torch.cat(transformed_data, dim=-1)

    # Convert to the desired output type if necessary
    if dtype is not None:
        transformed_data = transformed_data.to(dtype=dtype)

    return transformed_data


class Custom3DApplyTransformToPoints(InvertibleTransform, Transform):
    """
    Transform points between image coordinates and world coordinates.
    The input coordinates are assumed to be in the shape (C, N, 2 or 3), where C represents the number of channels
    and N denotes the number of points. It will return a tensor with the same shape as the input.

    Args:
        dtype: The desired data type for the output.
        affine: A 3x3 or 4x4 affine transformation matrix applied to points. This matrix typically originates
            from the image. For 2D points, a 3x3 matrix can be provided, avoiding the need to add an unnecessary
            Z dimension. While a 4x4 matrix is required for 3D transformations, it's important to note that when
            applying a 4x4 matrix to 2D points, the additional dimensions are handled accordingly.
            The matrix is always converted to float64 for computation, which can be computationally
            expensive when applied to a large number of points.
            If None, will try to use the affine matrix from the input data.
        invert_affine: Whether to invert the affine transformation matrix applied to the points. Defaults to ``True``.
            Typically, the affine matrix is derived from an image and represents its location in world space,
            while the points are in world coordinates. A value of ``True`` represents transforming these
            world space coordinates to the image's coordinate space, and ``False`` the inverse of this operation.
        affine_lps_to_ras: Defaults to ``False``. Set to `True` if your point data is in the RAS coordinate system
            or you're using `ITKReader` with `affine_lps_to_ras=True`.
            This ensures the correct application of the affine transformation between LPS (left-posterior-superior)
            and RAS (right-anterior-superior) coordinate systems. This argument ensures the points and the affine
            matrix are in the same coordinate system.

    Use Cases:
        - Transforming points between world space and image space, and vice versa.
        - Automatically handling inverse transformations between image space and world space.
        - If points have an existing affine transformation, the class computes and
          applies the required delta affine transformation.

    """

    def __init__(
        self,
        dtype: DtypeLike | torch.dtype | None = None,
        affine: torch.Tensor | None = None,
        invert_affine: bool = True,
        affine_lps_to_ras: bool = False,
    ) -> None:
        self.dtype = dtype
        self.affine = affine
        self.invert_affine = invert_affine
        self.affine_lps_to_ras = affine_lps_to_ras

    def _compute_final_affine(
        self, affine: torch.Tensor, applied_affine: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute the final affine transformation matrix to apply to the point data.

        Args:
            data: Input coordinates assumed to be in the shape (C, N, 2 or 3).
            affine: 3x3 or 4x4 affine transformation matrix.

        Returns:
            Final affine transformation matrix.
        """

        affine = convert_data_type(affine, dtype=torch.float64)[0]

        if self.affine_lps_to_ras:
            affine = orientation_ras_lps(affine)

        if self.invert_affine:
            affine = linalg_inv(affine)
            if applied_affine is not None:
                affine = affine @ applied_affine

        return affine

    def transform_coordinates(
        self, data: torch.Tensor, affine: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Transform coordinates using an affine transformation matrix.

        Args:
            data: The input coordinates are assumed to be in the shape (C, N, 2 or 3),
                where C represents the number of channels and N denotes the number of points.
            affine: 3x3 or 4x4 affine transformation matrix. The matrix is always converted to float64 for computation,
                which can be computationally expensive when applied to a large number of points.

        Returns:
            Transformed coordinates.
        """
        data = convert_to_tensor(data, track_meta=get_track_meta())
        if affine is None and self.invert_affine:
            raise ValueError("affine must be provided when invert_affine is True.")
        # applied_affine is the affine transformation matrix that has already been applied to the point data
        applied_affine: torch.Tensor | None = getattr(data, "affine", None)
        affine = applied_affine if affine is None else affine
        if affine is None:
            raise ValueError(
                "affine must be provided if data does not have an affine matrix."
            )

        final_affine = self._compute_final_affine(affine, applied_affine)
        out = custom_apply_affine_to_keypoints_3d(data, final_affine, dtype=self.dtype)

        extra_info = {
            "invert_affine": self.invert_affine,
            "dtype": get_dtype_string(self.dtype),
            "image_affine": affine,
            "affine_lps_to_ras": self.affine_lps_to_ras,
        }

        xform = (
            orientation_ras_lps(linalg_inv(final_affine))
            if self.affine_lps_to_ras
            else linalg_inv(final_affine)
        )
        meta_info = TraceableTransform.track_transform_meta(
            data,
            affine=xform,
            extra_info=extra_info,
            transform_info=self.get_transform_info(),
        )

        return out, meta_info

    def __call__(self, data: torch.Tensor, affine: torch.Tensor | None = None):
        """
        Args:
            data: The input coordinates are assumed to be in the shape (C, N, 2 or 3),
                where C represents the number of channels and N denotes the number of points.
            affine: A 3x3 or 4x4 affine transformation matrix, this argument will take precedence over ``self.affine``.
        """
        if data.ndim != 3 or data.shape[-1] < 6:
            raise ValueError(
                f"data should be in shape (C, N, K) with K>=6, got {data.shape}."
            )
        affine = self.affine if affine is None else affine
        if affine is not None and affine.shape not in ((3, 3), (4, 4)):
            raise ValueError(
                f"affine should be in shape (3, 3) or (4, 4), got {affine.shape}."
            )

        out, meta_info = self.transform_coordinates(data, affine)

        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        inverse_transform = Custom3DApplyTransformToPoints(
            dtype=transform[TraceKeys.EXTRA_INFO]["dtype"],
            invert_affine=not transform[TraceKeys.EXTRA_INFO]["invert_affine"],
            affine_lps_to_ras=transform[TraceKeys.EXTRA_INFO]["affine_lps_to_ras"],
        )
        with inverse_transform.trace_transform(False):
            data = inverse_transform(
                data, transform[TraceKeys.EXTRA_INFO]["image_affine"]
            )

        return data


class Custom3DApplyTransformToPointsd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ApplyTransformToPoints`.
    The input coordinates are assumed to be in the shape (C, N, 2 or 3),
    where C represents the number of channels and N denotes the number of points.
    The output has the same shape as the input.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        refer_keys: The key of the reference item used for transformation.
            It can directly refer to an affine or an image from which the affine can be derived. It can also be a
            sequence of keys, in which case each refers to the affine applied to the matching points in `keys`.
        dtype: The desired data type for the output.
        affine: A 3x3 or 4x4 affine transformation matrix applied to points. This matrix typically originates
            from the image. For 2D points, a 3x3 matrix can be provided, avoiding the need to add an unnecessary
            Z dimension. While a 4x4 matrix is required for 3D transformations, it's important to note that when
            applying a 4x4 matrix to 2D points, the additional dimensions are handled accordingly.
            The matrix is always converted to float64 for computation, which can be computationally
            expensive when applied to a large number of points.
            If None, will try to use the affine matrix from the refer data.
        invert_affine: Whether to invert the affine transformation matrix applied to the points. Defaults to ``True``.
            Typically, the affine matrix is derived from the image, while the points are in world coordinates.
            If you want to align the points with the image, set this to ``True``. Otherwise, set it to ``False``.
        affine_lps_to_ras: Defaults to ``False``. Set to `True` if your point data is in the RAS coordinate system
            or you're using `ITKReader` with `affine_lps_to_ras=True`.
            This ensures the correct application of the affine transformation between LPS (left-posterior-superior)
            and RAS (right-anterior-superior) coordinate systems. This argument ensures the points and the affine
            matrix are in the same coordinate system.
        allow_missing_keys: Don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        refer_keys: KeysCollection | None = None,
        dtype: DtypeLike | torch.dtype = torch.float64,
        affine: torch.Tensor | None = None,
        invert_affine: bool = True,
        affine_lps_to_ras: bool = False,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.refer_keys = ensure_tuple_rep(refer_keys, len(self.keys))
        self.converter = Custom3DApplyTransformToPoints(
            dtype=dtype,
            affine=affine,
            invert_affine=invert_affine,
            affine_lps_to_ras=affine_lps_to_ras,
        )

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        d = dict(data)
        for key, refer_key in self.key_iterator(d, self.refer_keys):
            coords = d[key]
            affine = None  # represents using affine given in constructor
            if refer_key is not None:
                if refer_key in d:
                    refer_data = d[refer_key]
                else:
                    raise KeyError(
                        f"The refer_key '{refer_key}' is not found in the data."
                    )

                # use the "affine" member of refer_data, or refer_data itself, as the affine matrix
                affine = getattr(refer_data, "affine", refer_data)
            d[key] = self.converter(coords, affine)
        return d

    def inverse(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter.inverse(d[key])
        return d


def apply_affine_to_3d_normal_distribution(
    data: torch.Tensor, affine: torch.Tensor, dtype: torch.dtype = torch.float64
):
    """
    Apply affine transformation to a list of 3D Normal distributions

    Args:
        data: Input data to apply affine transformation, should be a tensor of shape (C, N, K),
            where N is the number of points, last dim has K >= 6 channels corresponding to:
                - First 3 channels: mean - (x, y, z) coordinates
                - Next 6 channels: covariance - represent by 6 values cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz.
                The 3x3 covariance matrix is
                    [
                        [cov_xx, cov_xy, cov_xz],
                        [cov_xy, cov_yy, cov_yz],
                        [cov_xz, cov_yz, cov_zz],
                    ]
        affine: Affine matrix to be applied, should be a tensor of shape (4, 4).
        dtype: Output data dtype (default: torch.float64).

    Returns:
        Tensor of transformed keypoints with the same shape as input.
    """
    assert len(data.shape) == 3
    data_: torch.Tensor = convert_to_tensor(data, track_meta=False, dtype=torch.float64)
    affine = to_affine_nd(3, affine)

    # Extract the 3D coordinates (first 3 channels)
    means = data_[:, :, :3]

    # Convert data to homogeneous coordinates (add the extra 1s for affine transformation)
    homogeneous_means = torch.cat(
        (
            means,
            torch.ones((means.shape[0], means.shape[1], 1), dtype=means.dtype),
        ),
        dim=2,
    )

    # Apply the affine transformation (matrix multiplication with affine transpose)
    transformed_means = torch.matmul(homogeneous_means, affine.T)

    # Extract the transformed coordinates (remove the homogeneous coordinate)
    transformed_means = transformed_means[:, :, :-1]

    transformed_data = [transformed_means]
    # Optionally, apply scaling from affine matrix (e.g., if affine includes scaling components)
    if data_.shape[2] >= 9:
        # [[0, 3, 4],
        #  [3, 1, 5],
        #  [4, 5, 2]]
        _cov_idxs = np.array([0, 3, 4, 3, 1, 5, 4, 5, 2]) + 3
        original_covs = data_[..., _cov_idxs].reshape(
            *data_.shape[:-1], 3, 3
        )  # (C,N,3,3)

        # Transform the covariance matrices
        A = affine[:3, :3]
        transformed_covs = torch.einsum(
            "ij,cnjk,kl->cnil", A, original_covs, A.T
        )  # (C,N,3,3)
        C, N, _, __ = transformed_covs.shape
        assert _ == __ == 3
        transformed_covs = transformed_covs.reshape(C, N, 9)
        transformed_covs = transformed_covs[:, :, [0, 4, 8, 1, 2, 5]]
        transformed_data.append(transformed_covs)
    if data_.shape[2] > 9:
        transformed_data.append(data_[:, :, 9:])
    # Concatenate the transformed coordinates with unchanged class index and transformed scales
    transformed_data = torch.cat(transformed_data, dim=-1)

    # Convert to the desired output type if necessary
    if dtype is not None:
        transformed_data = transformed_data.to(dtype=dtype)

    return transformed_data


class ApplyTransformToNormalDistributions(InvertibleTransform, Transform):
    """
    Transform points between image coordinates and world coordinates.
    The input coordinates are assumed to be in the shape (C, N, D),
    where C represents the number of channels and N denotes the
    number of points, D >= 9 (mean: 3, cov: 6).
    It will return a tensor with the same shape as the input.

    Args:
        dtype: The desired data type for the output.
        affine: A 3x3 or 4x4 affine transformation matrix applied to points. This matrix typically originates
            from the image. For 2D points, a 3x3 matrix can be provided, avoiding the need to add an unnecessary
            Z dimension. While a 4x4 matrix is required for 3D transformations, it's important to note that when
            applying a 4x4 matrix to 2D points, the additional dimensions are handled accordingly.
            The matrix is always converted to float64 for computation, which can be computationally
            expensive when applied to a large number of points.
            If None, will try to use the affine matrix from the input data.
        invert_affine: Whether to invert the affine transformation matrix applied to the points. Defaults to ``True``.
            Typically, the affine matrix is derived from an image and represents its location in world space,
            while the points are in world coordinates. A value of ``True`` represents transforming these
            world space coordinates to the image's coordinate space, and ``False`` the inverse of this operation.
        affine_lps_to_ras: Defaults to ``False``. Set to `True` if your point data is in the RAS coordinate system
            or you're using `ITKReader` with `affine_lps_to_ras=True`.
            This ensures the correct application of the affine transformation between LPS (left-posterior-superior)
            and RAS (right-anterior-superior) coordinate systems. This argument ensures the points and the affine
            matrix are in the same coordinate system.

    Use Cases:
        - Transforming points between world space and image space, and vice versa.
        - Automatically handling inverse transformations between image space and world space.
        - If points have an existing affine transformation, the class computes and
          applies the required delta affine transformation.

    """

    def __init__(
        self,
        dtype: DtypeLike | torch.dtype | None = None,
        affine: torch.Tensor | None = None,
        invert_affine: bool = True,
        affine_lps_to_ras: bool = False,
    ) -> None:
        self.dtype = dtype
        self.affine = affine
        self.invert_affine = invert_affine
        self.affine_lps_to_ras = affine_lps_to_ras

    def _compute_final_affine(
        self, affine: torch.Tensor, applied_affine: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute the final affine transformation matrix to apply to the point data.

        Args:
            data: Input coordinates assumed to be in the shape (C, N, 2 or 3).
            affine: 3x3 or 4x4 affine transformation matrix.

        Returns:
            Final affine transformation matrix.
        """

        affine = convert_data_type(affine, dtype=torch.float64)[0]

        if self.affine_lps_to_ras:
            affine = orientation_ras_lps(affine)

        if self.invert_affine:
            affine = linalg_inv(affine)
            if applied_affine is not None:
                affine = affine @ applied_affine

        return affine

    def transform_distributions(
        self, data: torch.Tensor, affine: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Transform coordinates using an affine transformation matrix.

        Args:
            data: The input coordinates are assumed to be in the shape (C, N, 2 or 3),
                where C represents the number of channels and N denotes the number of points.
            affine: 3x3 or 4x4 affine transformation matrix. The matrix is always converted to float64 for computation,
                which can be computationally expensive when applied to a large number of points.

        Returns:
            Transformed coordinates.
        """
        data = convert_to_tensor(data, track_meta=get_track_meta())
        assert data.shape[-1] >= 9
        if affine is None and self.invert_affine:
            raise ValueError("affine must be provided when invert_affine is True.")
        # applied_affine is the affine transformation matrix that has already been applied to the point data
        applied_affine: torch.Tensor | None = getattr(data, "affine", None)
        affine = applied_affine if affine is None else affine
        if affine is None:
            raise ValueError(
                "affine must be provided if data does not have an affine matrix."
            )

        final_affine = self._compute_final_affine(affine, applied_affine)
        out = apply_affine_to_3d_normal_distribution(
            data, final_affine, dtype=self.dtype
        )

        extra_info = {
            "invert_affine": self.invert_affine,
            "dtype": get_dtype_string(self.dtype),
            "image_affine": affine,
            "affine_lps_to_ras": self.affine_lps_to_ras,
        }

        xform = (
            orientation_ras_lps(linalg_inv(final_affine))
            if self.affine_lps_to_ras
            else linalg_inv(final_affine)
        )
        meta_info = TraceableTransform.track_transform_meta(
            data,
            affine=xform,
            extra_info=extra_info,
            transform_info=self.get_transform_info(),
        )

        return out, meta_info

    def __call__(self, data: torch.Tensor, affine: torch.Tensor | None = None):
        """
        Args:
            data: The input coordinates are assumed to be in the shape (C, N, 2 or 3),
                where C represents the number of channels and N denotes the number of points.
            affine: A 3x3 or 4x4 affine transformation matrix, this argument will take precedence over ``self.affine``.
        """
        if data.ndim != 3 or data.shape[-1] < 6:
            raise ValueError(
                f"data should be in shape (C, N, K) with K>=6, got {data.shape}."
            )
        affine = self.affine if affine is None else affine
        if affine is not None and affine.shape not in ((3, 3), (4, 4)):
            raise ValueError(
                f"affine should be in shape (3, 3) or (4, 4), got {affine.shape}."
            )

        out, meta_info = self.transform_distributions(data, affine)

        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        inverse_transform = ApplyTransformToNormalDistributions(
            dtype=transform[TraceKeys.EXTRA_INFO]["dtype"],
            invert_affine=not transform[TraceKeys.EXTRA_INFO]["invert_affine"],
            affine_lps_to_ras=transform[TraceKeys.EXTRA_INFO]["affine_lps_to_ras"],
        )
        with inverse_transform.trace_transform(False):
            data = inverse_transform(
                data, transform[TraceKeys.EXTRA_INFO]["image_affine"]
            )

        return data


class ApplyTransformToNormalDistributionsd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ApplyTransformToPoints`.
    The input coordinates are assumed to be in the shape (C, N, 2 or 3),
    where C represents the number of channels and N denotes the number of points.
    The output has the same shape as the input.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        refer_keys: The key of the reference item used for transformation.
            It can directly refer to an affine or an image from which the affine can be derived. It can also be a
            sequence of keys, in which case each refers to the affine applied to the matching points in `keys`.
        dtype: The desired data type for the output.
        affine: A 3x3 or 4x4 affine transformation matrix applied to points. This matrix typically originates
            from the image. For 2D points, a 3x3 matrix can be provided, avoiding the need to add an unnecessary
            Z dimension. While a 4x4 matrix is required for 3D transformations, it's important to note that when
            applying a 4x4 matrix to 2D points, the additional dimensions are handled accordingly.
            The matrix is always converted to float64 for computation, which can be computationally
            expensive when applied to a large number of points.
            If None, will try to use the affine matrix from the refer data.
        invert_affine: Whether to invert the affine transformation matrix applied to the points. Defaults to ``True``.
            Typically, the affine matrix is derived from the image, while the points are in world coordinates.
            If you want to align the points with the image, set this to ``True``. Otherwise, set it to ``False``.
        affine_lps_to_ras: Defaults to ``False``. Set to `True` if your point data is in the RAS coordinate system
            or you're using `ITKReader` with `affine_lps_to_ras=True`.
            This ensures the correct application of the affine transformation between LPS (left-posterior-superior)
            and RAS (right-anterior-superior) coordinate systems. This argument ensures the points and the affine
            matrix are in the same coordinate system.
        allow_missing_keys: Don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        refer_keys: KeysCollection | None = None,
        dtype: DtypeLike | torch.dtype = torch.float64,
        affine: torch.Tensor | None = None,
        invert_affine: bool = True,
        affine_lps_to_ras: bool = False,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.refer_keys = ensure_tuple_rep(refer_keys, len(self.keys))
        self.converter = ApplyTransformToNormalDistributions(
            dtype=dtype,
            affine=affine,
            invert_affine=invert_affine,
            affine_lps_to_ras=affine_lps_to_ras,
        )

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        d = dict(data)
        for key, refer_key in self.key_iterator(d, self.refer_keys):
            coords = d[key]
            affine = None  # represents using affine given in constructor
            if refer_key is not None:
                if refer_key in d:
                    refer_data = d[refer_key]
                else:
                    raise KeyError(
                        f"The refer_key '{refer_key}' is not found in the data."
                    )

                # use the "affine" member of refer_data, or refer_data itself, as the affine matrix
                affine = getattr(refer_data, "affine", refer_data)
            d[key] = self.converter(coords, affine)
        return d

    def inverse(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter.inverse(d[key])
        return d


from monai.transforms import Crop


class CropBySlicesd(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of abstract class :py:class:`monai.transforms.Crop`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        cropper: crop transform for the input image.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    backend = Crop.backend

    def __init__(
        self,
        keys: KeysCollection,
        slices_key: str,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy)
        self.cropper = Crop(lazy=lazy)
        self.slices_key = slices_key

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        self._lazy = value
        if isinstance(self.cropper, LazyTransform):
            self.cropper.lazy = value

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        slices = data[self.slices_key]
        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            d[key] = self.cropper(d[key], slices=slices, lazy=lazy_)  # type: ignore
        return d

    def inverse(
        self, data: Mapping[Hashable, MetaTensor]
    ) -> dict[Hashable, MetaTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.cropper.inverse(d[key])
        return d


class CustomMixer(RandomizableTransform):

    def __init__(self, batch_size: int, prob=1.0, alpha: float = 1.0) -> None:
        """
        Mixer is a base class providing the basic logic for the mixup-class of
        augmentations. In all cases, we need to sample the mixing weights for each
        sample (lambda in the notation used in the papers). Also, pairs of samples
        being mixed are picked by randomly shuffling the batch samples.

        Args:
            batch_size (int): number of samples per batch. That is, samples are expected tp
                be of size batchsize x channels [x depth] x height x width.
            alpha (float, optional): mixing weights are sampled from the Beta(alpha, alpha)
                distribution. Defaults to 1.0, the uniform distribution.
        """
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"Expected positive number, but got {alpha = }")
        self.prob = prob
        self.alpha = alpha
        self.batch_size = batch_size

    @abstractmethod
    def apply(self, data: torch.Tensor):
        raise NotImplementedError()

    def randomize(self, data=None) -> None:
        """
        Sometimes you need may to apply the same transform to different tensors.
        The idea is to get a sample and then apply it with apply() as often
        as needed. You need to call this method everytime you apply the transform to a new
        batch.
        """
        super().randomize(None)
        self._params = (
            torch.from_numpy(self.R.rand(self.batch_size) < self.prob),
            torch.from_numpy(self.R.beta(self.alpha, self.alpha, self.batch_size)).type(
                torch.float32
            ),
            self.R.permutation(self.batch_size),
            (
                [
                    torch.from_numpy(self.R.randint(0, d, size=(1,)))
                    for d in data.shape[2:]
                ]
                if data is not None
                else []
            ),
        )


class CustomMixUp(CustomMixer):
    """MixUp as described in:
    Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz.
    mixup: Beyond Empirical Risk Minimization, ICLR 2018

    Class derived from :py:class:`monai.transforms.Mixer`. See corresponding
    documentation for details on the constructor parameters.
    """

    def apply(self, data: torch.Tensor):
        do, weight, perm, _ = self._params
        weight[~do] = 1.0
        # print(do, weight, perm)
        nsamples, *dims = data.shape
        if len(weight) != nsamples:
            raise ValueError(
                f"Expected batch of size: {len(weight)}, but got {nsamples}"
            )

        if len(dims) not in [3, 4]:
            raise ValueError("Unexpected number of dimensions")

        mixweight = weight[(Ellipsis,) + (None,) * len(dims)]
        return mixweight * data + (1 - mixweight) * data[perm, ...]

    def apply_to_labels(self, labels, mode="mix"):
        if mode == "mix":
            return self.apply(labels)
        elif mode == "max":
            do, weight, perm, _ = self._params
            labels[do] = torch.maximum(labels[do], labels[perm[do]])
            return labels
        else:
            raise ValueError

    def __call__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor | None = None,
        randomize=True,
        label_mode="mix",
    ):
        data_t = convert_to_tensor(data, track_meta=get_track_meta())
        labels_t = data_t  # will not stay this value, needed to satisfy pylint/mypy
        if labels is not None:
            labels_t = convert_to_tensor(labels, track_meta=get_track_meta())
        if randomize:
            self.randomize()
        if labels is None:
            return convert_to_dst_type(self.apply(data_t), dst=data)[0]

        return (
            convert_to_dst_type(self.apply(data_t), dst=data)[0],
            convert_to_dst_type(
                self.apply_to_labels(labels_t, mode=label_mode), dst=labels
            )[0],
        )


class MixUpFromGenerator(RandomizableTransform):
    """MixUp as described in:
    Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz.
    mixup: Beyond Empirical Risk Minimization, ICLR 2018
    """

    def __init__(self, generator, prob=0.0, alpha=1.0, target_mode="mix"):
        super().__init__(prob=prob)
        self.alpha = alpha
        self.generator = generator
        self.target_mode = target_mode

    def apply(self, data1, data2, weight):
        return weight * data1 + (1 - weight) * data2

    def apply_to_target(self, target1, target2, weight, mode="mix"):
        if mode == "mix":
            return self.apply(target1, target2, weight)
        elif mode == "max":
            return torch.maximum(target1, target2)
        elif callable(mode):
            return mode(target1, target2, weight)
        else:
            raise ValueError

    def __call__(self, image, heatmap, label, kpts, randomize=True):
        # random select one target
        if randomize:
            self.randomize()
        if self._do_transform:
            image2, heatmap2, label2, kpts2 = next(self.generator)
            image_ret = self.apply(image, image2, weight=self._lambda)
            heatmap_ret = self.apply_to_target(
                heatmap, heatmap2, weight=self._lambda, mode=self.target_mode
            )
            label_ret = self.apply_to_target(
                label, label2, weight=self._lambda, mode=self.target_mode
            )
            kpts_ret = torch.cat([kpts, kpts2], dim=0)
            return image_ret, heatmap_ret, label_ret, kpts_ret
        else:
            return image, heatmap, label, kpts

    def randomize(self, data=None) -> None:
        super().randomize(None)
        self._lambda = float(self.R.beta(self.alpha, self.alpha, 1)[0])


class CutmixFromGenerator(RandomizableTransform):

    def __init__(self, generator, prob=0.0, alpha=1.0, inplace=False):
        super().__init__(prob=prob)
        self.alpha = alpha
        self.generator = generator
        self.inplace = inplace

    def apply(self, data1, data2, weight, coords, inplace=True):
        assert data1.shape == data2.shape
        if inplace:
            ret = data1
        else:
            ret = data1.clone()
        C, *dims = ret.shape
        lengths = [d * sqrt(1 - weight) for d in dims]
        cutmix_slices = [slice(None)] + [
            slice(c, min(ceil(c + ln), d)) for c, ln, d in zip(coords, lengths, dims)
        ]
        ret[cutmix_slices] = data2[cutmix_slices]
        return ret

    def __call__(self, image, heatmap, label, kpts, randomize=True):
        # random select one target
        if randomize:
            self.randomize(image)
        if self._do_transform:
            image2, heatmap2, label2, kpts2 = next(self.generator)
            if label2 is not None or kpts2 is not None:
                raise NotImplementedError
            image_ret = self.apply(
                image,
                image2,
                weight=self._lambda,
                coords=self._coords,
                inplace=self.inplace,
            )
            heatmap_ret = self.apply(
                heatmap,
                heatmap2,
                weight=self._lambda,
                coords=self._coords,
                inplace=self.inplace,
            )
            return image_ret, heatmap_ret, label, kpts
        else:
            return image, heatmap, label, kpts

    def randomize(self, data=None) -> None:
        super().randomize(None)
        self._lambda = float(self.R.beta(self.alpha, self.alpha, 1)[0])
        self._coords = (
            [torch.from_numpy(self.R.randint(0, d, size=(1,))) for d in data.shape[1:]]
            if data is not None
            else []
        )
        # print('randomize:', self._lambda, self._coords)


class CustomSpacingd(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Spacing`.

    This transform assumes the ``data`` dictionary has a key for the input
    data's metadata and contains `affine` field.  The key is formed by ``key_{meta_key_postfix}``.

    After resampling the input array, this transform will write the new affine
    to the `affine` field of metadata which is formed by ``key_{meta_key_postfix}``.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    see also:
        :py:class:`monai.transforms.Spacing`
    """

    backend = Spacing.backend

    def __init__(
        self,
        keys: KeysCollection,
        spacing_key: str,
        diagonal: bool = False,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.BORDER,
        align_corners: Sequence[bool] | bool = False,
        dtype: Sequence[DtypeLike] | DtypeLike = np.float64,
        scale_extent: bool = False,
        recompute_affine: bool = False,
        min_pixdim: Sequence[float] | float | None = None,
        max_pixdim: Sequence[float] | float | None = None,
        ensure_same_shape: bool = True,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            diagonal: whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::

                    np.diag((pixdim_0, pixdim_1, pixdim_2, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, the axes orientation, orthogonal rotation and
                translations components from the original affine will be
                preserved in the target affine. This option will not flip/swap
                axes against the original ones.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                It also can be a sequence of bool, each element corresponds to a key in ``keys``.
            dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
                It also can be a sequence of dtypes, each element corresponds to a key in ``keys``.
            scale_extent: whether the scale is computed based on the spacing or the full extent of voxels,
                default False. The option is ignored if output spatial size is specified when calling this transform.
                See also: :py:func:`monai.data.utils.compute_shape_offset`. When this is True, `align_corners`
                should be `True` because `compute_shape_offset` already provides the corner alignment shift/scaling.
            recompute_affine: whether to recompute affine based on the output shape. The affine computed
                analytically does not reflect the potential quantization errors in terms of the output shape.
                Set this flag to True to recompute the output affine based on the actual pixdim. Default to ``False``.
            min_pixdim: minimal input spacing to be resampled. If provided, input image with a larger spacing than this
                value will be kept in its original spacing (not be resampled to `pixdim`). Set it to `None` to use the
                value of `pixdim`. Default to `None`.
            max_pixdim: maximal input spacing to be resampled. If provided, input image with a smaller spacing than this
                value will be kept in its original spacing (not be resampled to `pixdim`). Set it to `None` to use the
                value of `pixdim`. Default to `None`.
            ensure_same_shape: when the inputs have the same spatial shape, and almost the same pixdim,
                whether to ensure exactly the same output spatial shape.  Default to True.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.spacing_key = spacing_key
        self.diagonal = diagonal
        self.recompute_affine = recompute_affine
        self.min_pixdim = min_pixdim
        self.max_pixdim = max_pixdim
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.scale_extent = ensure_tuple_rep(scale_extent, len(self.keys))
        self.ensure_same_shape = ensure_same_shape

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None
    ) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d: dict = dict(data)

        spacing = d[self.spacing_key]
        spacing_transform = Spacing(
            pixdim=spacing,
            diagonal=self.diagonal,
            recompute_affine=self.recompute_affine,
            min_pixdim=self.min_pixdim,
            max_pixdim=self.max_pixdim,
            lazy=self._lazy,
        )

        _init_shape, _pixdim, should_match = None, None, False
        output_shape_k = None  # tracking output shape
        lazy_ = self.lazy if lazy is None else lazy

        for (
            key,
            mode,
            padding_mode,
            align_corners,
            dtype,
            scale_extent,
        ) in self.key_iterator(
            d,
            self.mode,
            self.padding_mode,
            self.align_corners,
            self.dtype,
            self.scale_extent,
        ):
            if self.ensure_same_shape and isinstance(d[key], MetaTensor):
                if _init_shape is None and _pixdim is None:
                    _init_shape, _pixdim = d[key].peek_pending_shape(), d[key].pixdim
                else:
                    should_match = np.allclose(
                        _init_shape, d[key].peek_pending_shape()
                    ) and np.allclose(_pixdim, d[key].pixdim, atol=1e-3)
            d[key] = spacing_transform(
                data_array=d[key],
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                dtype=dtype,
                scale_extent=scale_extent,
                output_spatial_shape=output_shape_k if should_match else None,
                lazy=lazy_,
            )
            if output_shape_k is None:
                output_shape_k = (
                    d[key].peek_pending_shape()
                    if isinstance(d[key], MetaTensor)
                    else d[key].shape[1:]
                )
        return d

    def inverse(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        raise NotImplementedError
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.spacing_transform.inverse(cast(torch.Tensor, d[key]))
        return d


def crop_func(
    img: torch.Tensor,
    slices: tuple[slice, ...],
    lazy: bool,
    transform_info: dict,
    keep_shape=False,
    is_custom=False,
) -> torch.Tensor:
    """
    Functional implementation of cropping a MetaTensor. This function operates eagerly or lazily according
    to ``lazy`` (default ``False``).

    Args:
        img: data to be transformed, assuming `img` is channel-first and cropping doesn't apply to the channel dim.
        slices: the crop slices computed based on specified `center & size` or `start & end` or `slices`.
        lazy: a flag indicating whether the operation should be performed in a lazy fashion or not.
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    img_size = (
        img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    )
    spatial_rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else 3
    cropped = np.asarray(
        [[s.indices(o)[0], o - s.indices(o)[1]] for s, o in zip(slices[1:], img_size)]
    )
    extra_info = {"cropped": cropped.flatten().tolist()}
    to_shift = []
    for i, s in enumerate(ensure_tuple(slices)[1:]):
        if s.start is not None:
            if is_custom:
                assert lazy
                to_shift.append(s.start if s.start < 0 else s.start)
            else:
                to_shift.append(img_size[i] + s.start if s.start < 0 else s.start)
        else:
            to_shift.append(0)
    if keep_shape:
        assert lazy
        shape = [s.stop - s.start for s in slices[1:]]
    else:
        shape = [
            s.indices(o)[1] - s.indices(o)[0] for s, o in zip(slices[1:], img_size)
        ]
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=shape,
        affine=create_translate(spatial_rank, to_shift),
        extra_info=extra_info,
        orig_size=img_size,
        transform_info=transform_info,
        lazy=lazy,
    )
    out = convert_to_tensor(
        img.as_tensor() if isinstance(img, MetaTensor) else img,
        track_meta=get_track_meta(),
    )
    if lazy:
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info  # type: ignore
    out = out[slices]
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out  # type: ignore


class RandSpatialCropByKeypoints(Randomizable, Crop):
    """
    Crop image with random size or specific size ROI. It can crop at a random position as center
    or at the image center. And allows to set the minimum and maximum size to limit the randomly generated ROI.

    Note: even `random_size=False`, if a dimension of the expected ROI size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected ROI, and the cropped results
    of several images may not have exactly the same shape.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        max_roi_size: if `random_size` is True and `roi_size` specifies the min crop region size, `max_roi_size`
            can specify the max crop region size. if None, defaults to the input image size.
            if its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            if True, the actual size is sampled from `randint(roi_size, max_roi_size + 1)`.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    def __init__(
        self,
        roi_size: Sequence[int] | int,
        max_roi_size: Sequence[int] | int | None = None,
        random_center: bool = True,
        random_size: bool = False,
        margin: int | float = 0,
        auto_correct_center=True,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self.roi_size = roi_size
        self.max_roi_size = max_roi_size
        self.random_center = random_center
        self.random_size = random_size
        if isinstance(margin, (tuple, list)):
            assert len(margin) == len(roi_size)
        else:
            margin = [margin for _ in range(len(roi_size))]
        if isinstance(margin[0], float):
            assert all([isinstance(e, float) for e in margin])
            margin = [round(m * s) for m, s in zip(margin, roi_size)]
        else:
            assert all([isinstance(e, int) for e in margin])
        assert all([2 * m < s for m, s in zip(margin, roi_size)])

        self.margin = margin
        self.auto_correct_center = auto_correct_center
        self._size: Sequence[int] | None = None
        self._slices: tuple[slice, ...]

    def randomize(self, img_size: Sequence[int], keypoints) -> None:
        self._size = fall_back_tuple(self.roi_size, img_size)
        if self.random_size:
            max_size = (
                img_size
                if self.max_roi_size is None
                else fall_back_tuple(self.max_roi_size, img_size)
            )
            if any(i > j for i, j in zip(self._size, max_size)):
                raise ValueError(
                    f"min ROI size: {self._size} is larger than max ROI size: {max_size}."
                )
            self._size = tuple(
                self.R.randint(low=self._size[i], high=max_size[i] + 1)
                for i in range(len(img_size))
            )
        assert len(self._size) == 3 and all([e > 0 for e in self._size])

        assert keypoints.shape[0] == 1
        keypoints = keypoints[0]
        if len(keypoints) > 0:
            kpt_idx = self.R.randint(0, len(keypoints))
            xyz = keypoints[kpt_idx][:3].tolist()
            # assert all([0 <= round(c) < s for c, s in zip(xyz, img_size)])
            if self.random_center:
                valid_start = [
                    math.ceil(c - ps // 2 + m)
                    for c, ps, m in zip(xyz, self._size, self.margin)
                ]
                valid_end = [
                    int(c + ps // 2 - m)
                    for c, ps, m in zip(xyz, self._size, self.margin)
                ]
                center = [
                    self.R.randint(_min, _max + 1)
                    for _min, _max in zip(valid_start, valid_end)
                ]
            else:
                # keypoint is at the center of cropped patch
                center = [round(c) for c in xyz]
        else:
            valid_start = [
                min(0, s - ps) + ps // 2 for s, ps in zip(img_size, self._size)
            ]
            valid_end = [
                max(0, s - ps) + ps // 2 for s, ps in zip(img_size, self._size)
            ]
            center = [
                self.R.randint(_min, _max + 1)
                for _min, _max in zip(valid_start, valid_end)
            ]
        if self.auto_correct_center:
            center = correct_crop_centers(
                center, self._size, img_size, allow_smaller=True
            )
        # allow negative and go outside of current image
        patch_start = [c - ps // 2 for c, ps in zip(center, self._size)]
        patch_end = [start + ps for start, ps in zip(patch_start, self._size)]
        # print(
        #     f"XYZ={xyz} CENTER={center} PATH_START={patch_start} PATCH_END={patch_end}"
        # )
        self._slices = [slice(start, end) for start, end in zip(patch_start, patch_end)]

    def __call__(self, img: torch.Tensor, keypoints: torch.Tensor | None = None, randomize: bool = True, lazy: bool | None = None) -> torch.Tensor:  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        img_size = (
            img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
        )
        if randomize:
            self.randomize(img_size, keypoints)
        if self._size is None:
            raise RuntimeError("self._size not specified.")
        lazy_ = self.lazy if lazy is None else lazy

        # return super().__call__(img=img, slices=self._slices, lazy=lazy_)

        slices_ = list(self._slices)
        sd = len(
            img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
        )  # spatial dims
        if len(slices_) < sd:
            slices_ += [slice(None)] * (sd - len(slices_))
        # Add in the channel (no cropping)
        slices_ = list([slice(None)] + slices_[:sd])

        img_t: MetaTensor = convert_to_tensor(data=img, track_meta=get_track_meta())
        ret = crop_func(
            img_t,
            tuple(slices_),
            lazy_,
            self.get_transform_info(),
            keep_shape=True,
            is_custom=True,
        )
        return ret


class RandSpatialCropByKeypointsd(Cropd, Randomizable):
    """
    Base class for random crop transform.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        cropper: random crop transform for the input image.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    backend = Cropd.backend

    def __init__(
        self,
        keys: KeysCollection,
        keypoints_key: str,
        roi_size: Sequence[int] | int,
        max_roi_size: Sequence[int] | int | None = None,
        random_center: bool = True,
        random_size: bool = False,
        margin: int | float = 0,
        auto_correct_center: bool = True,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ):
        self.keypoints_key = keypoints_key
        cropper = RandSpatialCropByKeypoints(
            roi_size,
            max_roi_size,
            random_center,
            random_size,
            margin,
            auto_correct_center,
            lazy,
        )
        super().__init__(
            keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandCropd:
        super().set_random_state(seed, state)
        if isinstance(self.cropper, Randomizable):
            self.cropper.set_random_state(seed, state)
        return self

    def randomize(self, img_size: Sequence[int], keypoints: torch.Tensor) -> None:
        if isinstance(self.cropper, Randomizable):
            self.cropper.randomize(img_size, keypoints)

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        # the first key must exist to execute random operations
        first_item = d[self.first_key(d)]
        keypoints = d[self.keypoints_key]
        self.randomize(
            (
                first_item.peek_pending_shape()
                if isinstance(first_item, MetaTensor)
                else first_item.shape[1:]
            ),
            keypoints,
        )
        lazy_ = self.lazy if lazy is None else lazy
        if lazy_ is True and not isinstance(self.cropper, LazyTrait):
            raise ValueError(
                "'self.cropper' must inherit LazyTrait if lazy is True "
                f"'self.cropper' is of type({type(self.cropper)}"
            )
        for key in self.key_iterator(d):
            kwargs = (
                {"randomize": False} if isinstance(self.cropper, Randomizable) else {}
            )
            if isinstance(self.cropper, LazyTrait):
                kwargs["lazy"] = lazy_
            d[key] = self.cropper(d[key], **kwargs)  # type: ignore
        return d


class RandCoarseTransformWithKeypoints(RandomizableTransform):
    """
    Randomly select coarse regions in the image, then execute transform operations for the regions.
    It's the base class of all kinds of region transforms.
    Refer to papers: https://arxiv.org/abs/1708.04552

    Args:
        holes: number of regions to dropout, if `max_holes` is not None, use this arg as the minimum number to
            randomly select the expected number of regions.
        spatial_size: spatial size of the regions to dropout, if `max_spatial_size` is not None, use this arg
            as the minimum spatial size to randomly select size for every region.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        max_holes: if not None, define the maximum number to randomly select the expected number of regions.
        max_spatial_size: if not None, define the maximum spatial size to randomly select size for every region.
            if some components of the `max_spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `max_spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        prob: probability of applying the transform.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        holes: int,
        spatial_size: Sequence[int] | int,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        remove: str = "none",
        keypoint_margins: Any = "auto",
        max_retries: int = 10,
        prob: float = 0.1,
    ) -> None:
        assert remove in ["none", "keypoint", "patch"]
        RandomizableTransform.__init__(self, prob)
        if holes < 1:
            raise ValueError("number of holes must be greater than 0.")
        self.holes = holes
        self.spatial_size = spatial_size
        self.max_holes = max_holes
        self.max_spatial_size = max_spatial_size
        self.remove = remove
        self.max_retries = max_retries
        self.keypoint_margins = keypoint_margins
        if keypoint_margins == "auto":
            pass
        elif isinstance(keypoint_margins, (int, float)):
            self.keypoint_margins = [keypoint_margins] * 3
        else:
            assert len(keypoint_margins) == 3
        self.hole_coords: list = []

    def get_random_patch(
        self,
        dims: Sequence[int],
        patch_size: Sequence[int],
        rand_state: np.random.RandomState | None = None,
    ) -> tuple[slice, ...]:
        """
        Returns a tuple of slices to define a random patch in an array of shape `dims` with size `patch_size` or the as
        close to it as possible within the given dimension. It is expected that `patch_size` is a valid patch for a source
        of shape `dims` as returned by `get_valid_patch_size`.

        Args:
            dims: shape of source array
            patch_size: shape of patch size to generate
            rand_state: a random state object to generate random numbers from

        Returns:
            (tuple of slice): a tuple of slice objects defining the patch
        """

        # choose the minimal corner of the patch
        rand_int = np.random.randint if rand_state is None else rand_state.randint
        min_corner = tuple(
            rand_int(0, ms - ps + 1) if ms > ps else 0
            for ms, ps in zip(dims, patch_size)
        )

        # create the slices for each dimension which define the patch in the source array
        return [[mc, mc + ps] for mc, ps in zip(min_corner, patch_size)]

    def check_keypoint_in_patch(self, keypoint, patch):
        if self.keypoint_margins == "auto":
            assert len(keypoint) == 10
            x, y, z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz, kpt_cls = keypoint
            cov_mat = torch.tensor(
                [
                    [cov_xx, cov_xy, cov_xz],
                    [cov_xy, cov_yy, cov_yz],
                    [cov_xz, cov_yz, cov_zz],
                ],
                dtype=torch.float32,
            )
            margins = cal_range_in_conf_interval(
                cov_mat, conf_interval=0.9, sigma_scale_factor=None
            ).tolist()
        else:
            margins = self.keypoint_margins
        assert len(patch) == 3
        return all(
            [
                start - m <= c <= end - 1 + m
                for c, (start, end), m in zip(keypoint[:3], patch, margins)
            ]
        )

    def randomize(self, img_size: Sequence[int], keypoints: NdarrayOrTensor) -> None:
        super().randomize(None)
        assert len(keypoints.shape) == 3 and keypoints.shape[0] == 1
        if not self._do_transform:
            return None
        size = fall_back_tuple(self.spatial_size, img_size)
        self.hole_coords = []  # clear previously computed coords
        num_holes = (
            self.holes
            if self.max_holes is None
            else self.R.randint(self.holes, self.max_holes + 1)
        )
        num_retries = self.max_retries
        while len(self.hole_coords) < num_holes:
            if self.max_spatial_size is not None:
                max_size = fall_back_tuple(self.max_spatial_size, img_size)
                size = tuple(
                    self.R.randint(low=size[i], high=max_size[i] + 1)
                    for i in range(len(img_size))
                )
            valid_size = get_valid_patch_size(img_size, size)
            patch = self.get_random_patch(img_size, valid_size, self.R)
            patch_slice = (slice(None),) + tuple(
                slice(start, end) for start, end in patch
            )
            if self.remove == "none":
                self.hole_coords.append(patch_slice)
            elif self.remove == "keypoint":
                num_spatial_dim = len(patch)
                assert num_spatial_dim == 3
                keep_idxs = []
                for kpt_idx, kpt in enumerate(keypoints[0]):
                    is_in = self.check_keypoint_in_patch(kpt, patch)
                    if not is_in:
                        keep_idxs.append(kpt_idx)
                    else:
                        pass
                keypoints = keypoints[:, keep_idxs]
                self.hole_coords.append(patch_slice)
            elif self.remove == "patch":
                is_invalid = any(
                    [self.check_keypoint_in_patch(kpt, patch) for kpt in keypoints[0]]
                )
                if is_invalid:
                    num_retries -= 1
                    if num_retries <= 0:
                        break
                    else:
                        pass
                else:
                    self.hole_coords.append(patch_slice)
                    num_retries = self.max_retries  # reset

        return self.hole_coords, keypoints

    @abstractmethod
    def _transform_holes(self, img: np.ndarray) -> np.ndarray:
        """
        Transform the randomly selected `self.hole_coords` in input images.

        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method."
        )

    def __call__(
        self,
        img: NdarrayOrTensor,
        keypoints: NdarrayOrTensor | None = None,
        randomize: bool = True,
    ) -> NdarrayOrTensor:
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            _, keypoints = self.randomize(img.shape[1:], keypoints)

        if not self._do_transform:
            return img, keypoints

        img_np, *_ = convert_data_type(img, np.ndarray)
        out = self._transform_holes(img=img_np)
        ret, *_ = convert_to_dst_type(src=out, dst=img)
        if keypoints is None:
            return ret
        else:
            return ret, keypoints


class RandCoarseDropoutWithKeypoints(RandCoarseTransformWithKeypoints):
    """
    Randomly coarse dropout regions in the image, then fill in the rectangular regions with specified value.
    Or keep the rectangular regions and fill in the other areas with specified value.
    Refer to papers: https://arxiv.org/abs/1708.04552, https://arxiv.org/pdf/1604.07379
    And other implementation: https://albumentations.ai/docs/api_reference/augmentations/transforms/
    #albumentations.augmentations.transforms.CoarseDropout.

    Args:
        holes: number of regions to dropout, if `max_holes` is not None, use this arg as the minimum number to
            randomly select the expected number of regions.
        spatial_size: spatial size of the regions to dropout, if `max_spatial_size` is not None, use this arg
            as the minimum spatial size to randomly select size for every region.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        dropout_holes: if `True`, dropout the regions of holes and fill value, if `False`, keep the holes and
            dropout the outside and fill value. default to `True`.
        fill_value: target value to fill the dropout regions, if providing a number, will use it as constant
            value to fill all the regions. if providing a tuple for the `min` and `max`, will randomly select
            value for every pixel / voxel from the range `[min, max)`. if None, will compute the `min` and `max`
            value of input image then randomly select value to fill, default to None.
        max_holes: if not None, define the maximum number to randomly select the expected number of regions.
        max_spatial_size: if not None, define the maximum spatial size to randomly select size for every region.
            if some components of the `max_spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `max_spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        prob: probability of applying the transform.

    """

    def __init__(
        self,
        holes: int,
        spatial_size: Sequence[int] | int,
        dropout_holes: bool = True,
        fill_value: tuple[float, float] | float | None = None,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        remove: str = "none",
        keypoint_margins: Any = "auto",
        max_retries: int = 10,
        prob: float = 0.1,
    ) -> None:
        super().__init__(
            holes=holes,
            spatial_size=spatial_size,
            max_holes=max_holes,
            max_spatial_size=max_spatial_size,
            remove=remove,
            keypoint_margins=keypoint_margins,
            max_retries=max_retries,
            prob=prob,
        )
        self.dropout_holes = dropout_holes
        if isinstance(fill_value, (tuple, list)):
            if len(fill_value) != 2:
                raise ValueError(
                    "fill value should contain 2 numbers if providing the `min` and `max`."
                )
        self.fill_value = fill_value

    def _transform_holes(self, img: np.ndarray):
        """
        Fill the randomly selected `self.hole_coords` in input images.
        Please note that we usually only use `self.R` in `randomize()` method, here is a special case.

        """
        fill_value = (
            (img.min(), img.max()) if self.fill_value is None else self.fill_value
        )

        if self.dropout_holes:
            for h in self.hole_coords:
                if isinstance(fill_value, (tuple, list)):
                    img[h] = self.R.uniform(
                        fill_value[0], fill_value[1], size=img[h].shape
                    )
                else:
                    img[h] = fill_value
            ret = img
        else:
            if isinstance(fill_value, (tuple, list)):
                ret = self.R.uniform(
                    fill_value[0], fill_value[1], size=img.shape
                ).astype(img.dtype, copy=False)
            else:
                ret = np.full_like(img, fill_value)
            for h in self.hole_coords:
                ret[h] = img[h]
        return ret


class RandCoarseDropoutWithKeypointsd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandCoarseDropout`.
    Expect all the data specified by `keys` have same spatial shape and will randomly dropout the same regions
    for every key, if want to dropout differently for every key, please use this transform separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        holes: number of regions to dropout, if `max_holes` is not None, use this arg as the minimum number to
            randomly select the expected number of regions.
        spatial_size: spatial size of the regions to dropout, if `max_spatial_size` is not None, use this arg
            as the minimum spatial size to randomly select size for every region.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        dropout_holes: if `True`, dropout the regions of holes and fill value, if `False`, keep the holes and
            dropout the outside and fill value. default to `True`.
        fill_value: target value to fill the dropout regions, if providing a number, will use it as constant
            value to fill all the regions. if providing a tuple for the `min` and `max`, will randomly select
            value for every pixel / voxel from the range `[min, max)`. if None, will compute the `min` and `max`
            value of input image then randomly select value to fill, default to None.
        max_holes: if not None, define the maximum number to randomly select the expected number of regions.
        max_spatial_size: if not None, define the maximum spatial size to randomly select size for every region.
            if some components of the `max_spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `max_spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        prob: probability of applying the transform.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = RandCoarseDropoutWithKeypoints.backend

    def __init__(
        self,
        keys: KeysCollection,
        keypoints_key: str,
        holes: int,
        spatial_size: Sequence[int] | int,
        dropout_holes: bool = True,
        fill_value: tuple[float, float] | float | None = None,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        remove: str = "none",
        keypoint_margins: Any = "auto",
        max_retries: int = 10,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.dropper = RandCoarseDropoutWithKeypoints(
            holes=holes,
            spatial_size=spatial_size,
            dropout_holes=dropout_holes,
            fill_value=fill_value,
            max_holes=max_holes,
            max_spatial_size=max_spatial_size,
            remove=remove,
            keypoint_margins=keypoint_margins,
            max_retries=max_retries,
            prob=1.0,
        )
        self.keypoints_key = keypoints_key

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ):
        super().set_random_state(seed, state)
        self.dropper.set_random_state(seed, state)
        return self

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        keypoints = d[self.keypoints_key]
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # expect all the specified keys have same spatial shape and share same random holes
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        _hole_coords, keypoints = self.dropper.randomize(
            d[first_key].shape[1:], keypoints
        )
        d[self.keypoints_key] = keypoints
        for key in self.key_iterator(d):
            d[key] = self.dropper(img=d[key], randomize=False)
        return d
