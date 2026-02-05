import logging
import math

import timm
import torch
from torch import nn
from yagm.backbones import BackboneMixin
from yagm.backbones.utils import (
    prune_timm_backbone,
    resolve_hf_preprocess_mean_std,
    resolve_timm_feature_channels,
    resolve_timm_feature_strides,
    resolve_timm_preprocess_mean_std,
)
from yagm.layers.common import StandardNormalize

logger = logging.getLogger(__name__)

__all__ = [
    "TimmBackbone",
    "ForwardFeaturesTimmBackbone",
    "ForwardIntermediatesTimmBackbone",
]


class TimmBackbone(BackboneMixin):

    def __init__(self, model_name, **kwargs):
        super().__init__()
        kwargs_output_strides = kwargs.pop("output_strides", None)
        features_only = kwargs.pop("features_only", True)
        pretrained = kwargs.pop("pretrained", True)
        _normalize_method = kwargs.pop("normalize", "mean_std")
        _norm_mean = kwargs.pop("mean", None)
        _norm_std = kwargs.pop("std", None)

        if not pretrained:
            logger.warning(
                "\n\n!!! pretrained=False, skip loading pretrained model !!!\n\n"
            )

        backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=features_only, **kwargs
        )
        prune_timm_backbone(backbone, prune_norm=False, prune_head=True)

        self._output_channels = resolve_timm_feature_channels(backbone)
        self._output_strides = resolve_timm_feature_strides(backbone)
        if self._output_strides is None:
            self._output_strides = kwargs_output_strides
        if kwargs_output_strides is not None:
            assert self._output_strides == kwargs_output_strides
        in_channels = kwargs.get("in_chans", 3)

        if _normalize_method == "mean_std":
            # Normalization params: mean, std
            if _norm_mean is None and _norm_std is None:
                # attempt extract from model
                mean, std = resolve_timm_preprocess_mean_std(backbone)
            elif _norm_mean is not None and _norm_std is not None:
                mean = torch.FloatTensor(mean)
                std = torch.FloatTensor(std)
            else:
                raise ValueError

            _C = mean.numel()
            assert _C == std.numel()
            mean = mean.reshape(1, _C, 1, 1).repeat(
                1, math.ceil(in_channels / _C), 1, 1
            )[:, :in_channels]
            std = std.reshape(1, _C, 1, 1).repeat(1, math.ceil(in_channels / _C), 1, 1)[
                :, :in_channels
            ]
            logger.info("Backbone normalization using\nMEAN=%s\nSTD=%s\n", mean, std)
            # keep the correct order, better interpretation
            self.normalize = StandardNormalize(mean, std, inplace=False)
        self._backbone = backbone
        self.in_channels = in_channels
        self._append_stride1_feature = self._output_strides[0] != 1

    @property
    def output_channels(self):
        if self._output_strides[0] == 1:
            return self._output_channels
        else:
            return [self.in_channels] + self._output_channels

    @property
    def output_strides(self):
        return (
            [1] + self._output_strides
            if self._output_strides[0] != 1
            else self._output_strides
        )

    def forward(self, x, *args, **kwargs):
        x = self.normalize(x)
        features = self._backbone(x, *args, **kwargs)
        if self._append_stride1_feature:
            return [x] + features
        else:
            return features


class ForwardFeaturesTimmBackbone(TimmBackbone):
    """
    Used in transformer variants such as Coat,..
    """

    def __init__(self, model_name, **kwargs):
        features_only = kwargs.pop("features_only", False)
        assert features_only == False
        super().__init__(model_name, features_only=features_only, **kwargs)

    def forward(self, x):
        x = self.normalize(x)
        features = self._backbone.forward_features(x)
        return [x] + list(features.values())


class ForwardIntermediatesTimmBackbone(TimmBackbone):
    """
    Most populars variants: ConvNeXt, Maxvit, Swin, EfficientNet,..
    """

    def __init__(self, model_name, **kwargs):
        features_only = kwargs.pop("features_only", False)
        assert features_only == False
        # usual 4-level strides: (4, 8, 16, 32)
        # some unusual archs where stride-2 feature map is available: Maxvit
        self._output_indices = kwargs.pop("output_indices", (0, 1, 2, 3))
        super().__init__(model_name, features_only=features_only, **kwargs)

    def forward(self, x):
        x = self.normalize(x)
        return [x] + self._backbone.forward_intermediates(
            x,
            indices=self._output_indices,
            norm=True,  # norm on last feature
            stop_early=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )
