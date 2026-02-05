import logging
import math
import random
import typing
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d

from yagm.transforms.onebumentations import functional
from yagm.transforms.onebumentations import utils
from yagm.transforms.onebumentations import BaseCompose
from yagm.transforms.onebumentations.interface import (
    BasicTransform,
    SequenceTransform,
    DualTransform,
    ScaleFloatType,
    to_tuple,
)

logger = logging.getLogger(__name__)

TransformType = typing.Union[BasicTransform, BaseCompose]
REPR_INDENT_STEP = 4

__all__ = ["TemporalResample"]


class _Template(DualTransform):

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, seq, **params):
        pass

    def get_params(self) -> Dict:
        return super().get_params()

    @property
    def targets_as_params(self) -> List[str]:
        return []

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def get_transform_init_args_names(self):
        return tuple()


class Flip(DualTransform):

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, seq, **params) -> np.ndarray:
        seq["seq"] = seq[::-1]
        return seq

    def apply_keypoints(self, keypoints, T, C, **params):
        # 0 -> T, 0.5 -> T - 0.5
        return T - keypoints

    def get_params(self) -> Dict:
        return super().get_params()

    def get_transform_init_args_names(self):
        return tuple()

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        T, C = params["seq"].shape
        return {"T": T, "C": C}


class TemporalResample(DualTransform):

    def __init__(
        self,
        scale_limit: Tuple[float, float] = (0.75, 1.25),
        mode="nearest",
        backend="scipy",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.scale_limit = scale_limit
        self.mode = mode
        self.backend = backend
        if mode == "random":
            self.mode_choices = functional.BACKEND_TO_INTERPOLATION_MODES[self.backend]

    def apply(self, seq, scale, mode, T, **params) -> np.ndarray:
        if scale == 1:
            return seq
        target_len = max(1, round(T * scale))
        ret = functional.interp_1d(seq, target_len, mode=mode, backend=self.backend)
        return ret

    def apply_keypoints(self, keypoints, scale, mode, T, **params):
        if scale == 1:
            return keypoints
        target_len = max(1, round(T * scale))
        return keypoints * target_len / T

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("scale_limit", "mode", "backend")

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        seq = params["seq"]
        T, C = seq.shape
        scale = random.uniform(self.scale_limit[0], self.scale_limit[1])
        if self.mode == "random":
            # scipy return nan-full array input contain just a single nan in spline interpolations: quadratic or cubic
            # https://github.com/scipy/scipy/issues/8781
            # https://github.com/scipy/scipy/issues/9315
            if self.backend == 'scipy' and np.isnan(seq).any():
                mode_choices = functional.SCIPY_CONTAIN_NAN_INTERPOLATION_MODES
            else:
                mode_choices = self.mode_choices
            mode = random.choice(mode_choices)
        else:
            mode = self.mode
        return {"scale": scale, "mode": mode, "T": T, "C": C}


class TemporalDoubleResample(DualTransform):
    """Usefull as A.Downscale() with more flexibility"""

    def __init__(
        self,
        scale_limit1: ScaleFloatType = 1.0,
        method1="nearest",
        scale_limit2: ScaleFloatType = 1.0,
        method2="nearest",
        backend="cv2",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.scale_limit1 = to_tuple(scale_limit1)
        self.method1 = method1
        self.scale_limit2 = to_tuple(scale_limit2)
        self.method2 = method2
        self.backend = backend

    def apply(self, seq, scale1, scale2, **params) -> np.ndarray:
        ori_seq_len = seq.shape[0]
        target_len1 = max(1, int(ori_seq_len * scale1))
        target_len2 = max(1, int(ori_seq_len * scale2))
        if target_len1 != ori_seq_len:
            seq = functional.interp_1d(
                seq, target_len1, mode=self.method1, backend=self.backend
            )
        if target_len2 != seq.shape[0]:
            seq = functional.interp_1d(
                seq, target_len2, mode=self.method2, backend=self.backend
            )
        new_seq_len = seq.shape[0]
        scale = (new_seq_len - 1) / (ori_seq_len - 1)
        seq["seq"] = seq
        seq["events"] = functional.scale_events(seq["events"], scale)
        return seq

    def get_params(self) -> Dict[str, float]:
        scale1 = random.uniform(self.scale_limit1[0], self.scale_limit1[1])
        scale2 = random.uniform(self.scale_limit2[0], self.scale_limit2[1])
        return {"scale1": scale1, "scale2": scale2}

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("scale_limit1", "method1", "scale_limit2", "method2", "backend")


class PerStageTransform(DualTransform):
    """This transform preserve label (events)"""

    def __init__(
        self,
        transform,
        transition_margin=5 * 12,
        stage_labels=[2, 3],
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.transform = transform
        self.transition_margin = round(transition_margin)
        self.stage_labels = stage_labels

    def apply(self, data, **params):
        seq = data["seq"]
        events = data["events"]
        stages = functional.split_to_stages(seq, events, self.transition_margin)
        segments = []
        for start, end, label in stages:
            ori_segment = seq[start:end]
            if label in self.stage_labels:
                _data = {"seq": ori_segment, "events": None}
                new_segment = self.transform(data=_data)["data"]["seq"]
                assert new_segment.shape[0] == ori_segment.shape[0]
                segments.append(new_segment)
            else:
                segments.append(ori_segment)
        if len(segments) > 1:
            new_seq = np.concatenate(segments, axis=0)
        else:
            new_seq = segments[0]
        data["seq"] = new_seq
        return data

    def get_params(self) -> Dict:
        return super().get_params()

    @property
    def targets_as_params(self) -> List[str]:
        return []

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def get_transform_init_args_names(self):
        return ("transform", "transition_margin")
