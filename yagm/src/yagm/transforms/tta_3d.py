from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor
from functools import lru_cache


def _extend_list(li, offset=0):
    ret = list(range(offset))
    ret.extend([e + offset for e in li])
    return ret


class BaseTTA:

    def __init__(self, ndim):
        self.ndim = ndim

    def transform_shape(self, shape: Tuple[int]) -> Tuple[int]:
        assert len(shape) >= self.ndim
        raise NotImplementedError

    def invert_shape(self, shape: Tuple[int]) -> Tuple[int]:
        assert len(shape) >= self.ndim
        raise NotImplementedError

    def transform_spacing(self, spacing: Tuple[float]) -> Tuple[float]:
        assert len(spacing) == self.ndim
        raise NotImplementedError

    def invert_spacing(self, spacing: Tuple[float]) -> Tuple[float]:
        assert len(spacing) == self.ndim
        raise NotImplementedError

    def transform(self, x: Tensor) -> Tuple[Tensor, Union[Tensor, float]]:
        raise NotImplementedError

    def invert(self, x: Tensor) -> Tuple[Tensor, Union[Tensor, float]]:
        raise NotImplementedError

    @lru_cache(maxsize=8)
    def get_transform_matrix(self, shape: Tuple[int]) -> Tensor:
        """Returns the transformation matrix (D+1)x(D+1) in homogeneous coordinates."""
        raise NotImplementedError

    def apply_transform_matrix(
        self, ori_kpts: torch.Tensor, matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply a (D+1)x(D+1) transformation matrix to (N,D) keypoints.
        """
        n, d = ori_kpts.shape
        assert d >= self.ndim
        assert matrix.shape == (self.ndim + 1, self.ndim + 1)
        ones = torch.ones((n, 1), dtype=ori_kpts.dtype, device=ori_kpts.device)
        kpts_h = torch.cat(
            [ori_kpts[:, : self.ndim], ones], dim=1
        )  # h: hormogeneous (N, D+1)
        # (D+1, D+1) @ (D+1, N) --> (N, D)
        transformed = (matrix @ kpts_h.T).T[:, : self.ndim]
        if ori_kpts.shape[1] > self.ndim:
            transformed = torch.cat([transformed, ori_kpts[:, self.ndim :]], dim=1)
        return transformed

    def transform_kpts(self, ori_kpts: Tensor, ori_shape: Tuple[int]) -> Tensor:
        M = self.get_transform_matrix(ori_shape[-self.ndim :])
        keep_handedness = np.linalg.det(M[: self.ndim, : self.ndim]) >= 0
        return keep_handedness, self.apply_transform_matrix(ori_kpts, M)

    def invert_kpts(self, ori_kpts: Tensor, ori_shape: Tuple[int]) -> Tensor:
        M = self.get_transform_matrix(ori_shape[-self.ndim :])
        M_inv = torch.inverse(M)
        keep_handedness = np.linalg.det(M_inv[: self.ndim, : self.ndim]) >= 0
        return keep_handedness, self.apply_transform_matrix(ori_kpts, M_inv)


class PermuteTTA(BaseTTA):
    """
    Simple dimensionals permuting using torch.permute()
    e.g, 6 possible permutations: xyz, xzy, yxz, yzx, zxy, zyx for 3D case

    Args:
        code: permutation ordering, relative to `ori_dims`
        ori_dims: notation of original dimension names
    """

    def __init__(self, code: str, weight: float, ori_dims="xyz"):
        ndim = len(ori_dims)
        super().__init__(ndim)
        permute_dims = [ori_dims.index(e) for e in code]
        assert sorted(permute_dims) == list(range(self.ndim))
        self.permute_dims = permute_dims
        self.inv_permute_dims = [permute_dims.index(e) for e in list(range(self.ndim))]
        self.code = code
        self.weight = weight
        self.ori_dims = ori_dims

    def __repr__(self):
        return f"{self.__class__.__name__}(code={self.code}, weight={self.weight}, ori_dims={self.ori_dims})"

    def transform_shape(self, shape: Tuple[int]) -> Tuple[int]:
        assert len(shape) >= self.ndim
        return tuple(shape[: -self.ndim]) + tuple(
            shape[-self.ndim :][d] for d in self.permute_dims
        )

    def invert_shape(self, shape: Tuple[int]) -> Tuple[int]:
        assert len(shape) == self.ndim
        return tuple(shape[: -self.ndim]) + tuple(
            shape[-self.ndim :][d] for d in self.inv_permute_dims
        )

    def transform_spacing(self, spacing: Tuple[float]) -> Tuple[float]:
        assert len(spacing) == self.ndim
        return tuple(spacing[d] for d in self.permute_dims)

    def invert_spacing(self, spacing: Tuple[float]) -> Tuple[float]:
        assert len(spacing) == self.ndim
        return tuple(spacing[d] for d in self.inv_permute_dims)

    def transform(self, x):
        offset = len(x.shape) - self.ndim
        assert offset >= 0
        x = torch.permute(x, _extend_list(self.permute_dims, offset))
        return x, self.weight

    def invert(self, x):
        offset = len(x.shape) - self.ndim
        assert offset >= 0
        x = torch.permute(x, _extend_list(self.inv_permute_dims, offset))
        return x, self.weight

    @lru_cache(8)
    def get_transform_matrix(self, shape: Tuple[int]) -> Tensor:
        del shape
        D = self.ndim
        M = torch.zeros((D + 1, D + 1))
        for i, j in enumerate(self.permute_dims):
            M[i, j] = 1
        M[-1, -1] = 1
        return M


class PermuteByRot90TTA(BaseTTA):
    """
    Use consecutive rot90() to archived shape specified by `code`,
    butkeep handedness (no flip operation).
    """

    def __init__(self, code: str, weight: float, ori_dims="xyz"):
        self.ndim = len(ori_dims)
        permute_dims = [ori_dims.index(e) for e in code]
        assert sorted(permute_dims) == list(range(self.ndim))
        self.permute_dims = permute_dims
        self.inv_permute_dims = [permute_dims.index(e) for e in list(range(self.ndim))]
        rot_params = self.min_swaps_planning(permute_dims)
        self.rot_params = rot_params
        self.code = code
        self.weight = weight
        self.ori_dims = ori_dims

    def __repr__(self):
        return f"{self.__class__.__name__}(code={self.code}, weight={self.weight}, ori_dims={self.ori_dims})"

    def min_swaps_planning(self, L):
        """
        Finds the minimum number of swaps to sort a list of consecutive numbers 0..N-1.

        Args:
            L: A list of N consecutive numbers 0..N-1 in random order.

        Returns:
            A list of tuples (i, j) representing the swap operations, or None if input is invalid.
            Returns empty list if the input is already sorted.
        """

        n = len(L)
        if n == 0:
            return []

        if not all(
            0 <= x < n for x in L
        ):  # check if all elements are in the range 0..N-1
            return None  # Or raise an exception: raise ValueError("Invalid input list: elements not in range 0..N-1")
        if len(set(L)) != n:
            return None  # check if all elements are distinct

        swaps = []
        visited = [False] * n

        for i in range(n):
            if visited[i] or L[i] == i:  # Already visited or in correct position
                continue

            cycle_size = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = L[j]  # No need to subtract 1 now
                cycle_size += 1

            if cycle_size > 1:  # Need swaps if cycle has more than one element
                j = i
                for _ in range(cycle_size - 1):
                    k = L[j]
                    swaps.append((j, k))
                    j = k
        return swaps

    def transform_shape(self, shape: Tuple[int]) -> Tuple[int]:
        assert len(shape) >= self.ndim
        return tuple(shape[: -self.ndim]) + tuple(
            shape[-self.ndim :][d] for d in self.permute_dims
        )

    def invert_shape(self, shape: Tuple[int]) -> Tuple[int]:
        assert len(shape) >= self.ndim
        return tuple(shape[: -self.ndim]) + tuple(
            shape[-self.ndim :][d] for d in self.inv_permute_dims
        )

    def transform_spacing(self, spacing: Tuple[float]) -> Tuple[float]:
        assert len(spacing) == self.ndim
        return tuple(spacing[d] for d in self.permute_dims)

    def invert_spacing(self, spacing: Tuple[float]) -> Tuple[float]:
        assert len(spacing) == self.ndim
        return tuple(spacing[d] for d in self.inv_permute_dims)

    def transform(self, x):
        offset = len(x.shape) - self.ndim
        assert offset >= 0
        for dims in self.rot_params:
            dims = [d + offset for d in dims]
            x = torch.rot90(x, k=1, dims=dims)
        return x, self.weight

    def invert(self, x):
        offset = len(x.shape) - self.ndim
        assert offset >= 0
        for dims in self.rot_params[::-1]:
            dims = [d + offset for d in dims]
            x = torch.rot90(x, k=3, dims=dims)
        return x, self.weight

    @lru_cache(8)
    def get_transform_matrix(self, shape: Tuple[int]) -> torch.Tensor:
        D = self.ndim
        M = torch.eye(D + 1)
        current_shape = list(shape)

        for d1, d2 in self.rot_params:
            R = torch.eye(D + 1)
            R[d1, d1] = 0
            R[d2, d2] = 0
            R[d1, d2] = -1
            R[d2, d1] = 1

            offset = current_shape[d2]
            R[d1, -1] = float(offset)
            M = R @ M
            current_shape[d1], current_shape[d2] = current_shape[d2], current_shape[d1]

        return M


class FlipTTA(BaseTTA):
    """
    Flip TTA: 8 possible Flipping: identity, flip along one of [x, y, z, xy, xz, yz, xyz]

    Args:
        code: code represent the flipping dims, relative to `ori_dims`
        ori_dims: notation of original dimension names
    """

    def __init__(self, code: str, weight: float, ori_dims="xyz"):
        ndim = len(ori_dims)
        super().__init__(ndim)
        flip_dims = [ori_dims.index(e) for e in code]
        assert len(flip_dims) == len(set(flip_dims))
        self.flip_dims = flip_dims
        self.code = code
        self.weight = weight
        self.ori_dims = ori_dims

    def transform_shape(self, shape: Tuple[int]) -> Tuple[int]:
        assert len(shape) >= self.ndim
        return shape

    def invert_shape(self, shape: Tuple[int]) -> Tuple[int]:
        assert len(shape) >= self.ndim
        return shape

    def transform_spacing(self, spacing: Tuple[float]) -> Tuple[float]:
        assert len(spacing) == self.ndim
        return spacing

    def invert_spacing(self, spacing: Tuple[float]) -> Tuple[float]:
        assert len(spacing) == self.ndim
        return spacing

    def transform(self, x):
        offset = len(x.shape) - self.ndim
        assert offset >= 0
        x = torch.flip(x, [d + offset for d in self.flip_dims])
        return x, self.weight

    def invert(self, x):
        offset = len(x.shape) - self.ndim
        assert offset >= 0
        x = torch.flip(x, [d + offset for d in self.flip_dims])
        return x, self.weight

    def __repr__(self):
        return f"{self.__class__.__name__}(code={self.code}, weight={self.weight}, ori_dims={self.ori_dims})"

    @lru_cache(8)
    def get_transform_matrix(self, shape: Tuple[int]) -> Tensor:
        D = self.ndim
        M = torch.eye(D + 1)
        for d in self.flip_dims:
            M[d, d] = -1
            M[d, -1] = shape[d]
        return M


class PermuteFlipTTA(BaseTTA):
    """
    Permute + Flip TTA:
    - 6 possible permutations (rot90): xyz, xzy, yxz, yzx, zxy, zyx
    - 8 possible Flipping: identity, flip along one of [x, y, z, xy, xz, yz, xyz]
    Total: 6 * 8 = 48 possible unique transformations

    Args:
        code: code represent the transform in form of `{permute_dims}_{flip_dims}`, e.g `zxy_xy`
        ori_dims: notation of original dimension names
    """

    def __init__(
        self, code: str, weight: float, ori_dims="xyz", permute_keep_handedness=True
    ):
        ndim = len(ori_dims)
        super().__init__(ndim)
        splits = code.split("_")
        if len(splits) == 1:
            permute_code = splits[0]
            flip_code = ""
        elif len(splits) == 2:
            permute_code, flip_code = splits
        else:
            raise ValueError

        if permute_keep_handedness:
            self.permute_tta = PermuteByRot90TTA(permute_code, weight, ori_dims)
        else:
            self.permute_tta = PermuteTTA(permute_code, weight, ori_dims)
        self.flip_tta = FlipTTA(flip_code, weight, ori_dims)
        self.code = code
        self.weight = weight
        self.ori_dims = ori_dims
        self.permute_keep_handedness = permute_keep_handedness

    def __repr__(self):
        return f"{self.__class__.__name__}(code={self.code}, weight={self.weight}, ori_dims={self.ori_dims}, permute_keep_handedness={self.permute_keep_handedness})"

    def transform_shape(self, shape: Tuple[int]) -> Tuple[int]:
        assert len(shape) >= self.ndim
        shape_after_perm = self.permute_tta.transform_shape(shape)
        shape_after_flip = self.flip_tta.transform_shape(shape_after_perm)
        return shape_after_flip

    def invert_shape(self, shape: Tuple[int]) -> Tuple[int]:
        assert len(shape) >= self.ndim
        shape_after_flip_inv = self.flip_tta.invert_shape(shape)
        shape_after_perm_inv = self.permute_tta.invert_shape(shape_after_flip_inv)
        return shape_after_perm_inv

    def transform_spacing(self, spacing: Tuple[float]) -> Tuple[float]:
        assert len(spacing) == self.ndim
        spacing_after_perm = self.permute_tta.transform_spacing(spacing)
        spacing_after_flip = self.flip_tta.transform_spacing(spacing_after_perm)
        return spacing_after_flip

    def invert_spacing(self, spacing: Tuple[float]) -> Tuple[float]:
        assert len(spacing) == self.ndim
        spacing_after_flip_inv = self.flip_tta.invert_spacing(spacing)
        spacing_after_perm_inv = self.permute_tta.invert_spacing(spacing_after_flip_inv)
        return spacing_after_perm_inv

    def transform(self, x):
        x, _ = self.permute_tta.transform(x)
        x, _ = self.flip_tta.transform(x)
        return x, self.weight

    def invert(self, x):
        x, _ = self.flip_tta.invert(x)
        x, _ = self.permute_tta.invert(x)
        return x, self.weight

    @lru_cache(8)
    def get_transform_matrix(self, shape: Tuple[int]) -> Tensor:
        M_perm = self.permute_tta.get_transform_matrix(shape)
        shape_after_perm = tuple(shape[i] for i in self.permute_tta.permute_dims)
        M_flip = self.flip_tta.get_transform_matrix(shape_after_perm)
        M = M_flip @ M_perm
        return M


def build_tta(code, weight, ori_dims="xyz", permute_keep_handedness=True):
    return PermuteFlipTTA(
        code, weight, ori_dims=ori_dims, permute_keep_handedness=permute_keep_handedness
    )
