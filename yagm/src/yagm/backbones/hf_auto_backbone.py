import logging
import math

import timm
import torch
from torch import nn
from transformers import AutoBackbone, TimmBackbone
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

__all__ = ["HFAutoBackbone"]


class HFAutoBackbone(BackboneMixin):
    """
    Idiot wrapper which support getting feature strides info :D
    """

    def __init__(self, model_name, **kwargs):
        super().__init__()
        kwargs_output_strides = kwargs.pop("output_strides", None)
        use_pretrained_backbone = kwargs.pop("pretrained", True)
        _normalize_method = kwargs.pop("normalize", "mean_std")
        _norm_mean = kwargs.pop("mean", None)
        _norm_std = kwargs.pop("std", None)
        if not use_pretrained_backbone:
            logger.warning(
                "\n\n!!! pretrained=False, skip loading pretrained model !!!\n\n"
            )
        hf_backbone = AutoBackbone.from_pretrained(
            model_name, use_pretrained_backbone=use_pretrained_backbone, **kwargs
        )
        self.config = hf_backbone.config

        if isinstance(hf_backbone, TimmBackbone):
            timm_output_strides = resolve_timm_feature_strides(hf_backbone._backbone)
            if timm_output_strides is not None:
                if kwargs_output_strides is not None:
                    assert (
                        timm_output_strides == kwargs_output_strides
                    ), "output_strides from kwargs is not matched with one from timm model"
                self._output_strides = timm_output_strides
            else:
                self._output_strides = kwargs_output_strides
        else:
            # @TODO - try to resolve from HF models as well
            self._output_strides = kwargs_output_strides

        if _normalize_method == "mean_std":
            # Normalization params: mean, std
            if _norm_mean is None and _norm_std is None:
                # attempt extract from model
                if isinstance(hf_backbone, TimmBackbone):
                    mean, std = resolve_timm_preprocess_mean_std(model)
                else:
                    mean, std = resolve_hf_preprocess_mean_std(model_name)
            elif _norm_mean is not None and _norm_std is not None:
                mean = torch.FloatTensor(mean)
                std = torch.FloatTensor(std)
            else:
                raise ValueError

            _C = mean.numel()
            assert _C == std.numel()
            in_channels = hf_backbone.config.num_channels
            mean = mean.reshape(1, _C, 1, 1).repeat(
                1, math.ceil(in_channels / _C), 1, 1
            )[:, :in_channels]
            std = std.reshape(1, _C, 1, 1).repeat(1, math.ceil(in_channels / _C), 1, 1)[
                :, :in_channels
            ]
            logger.info("Backbone normalization using\nMEAN=%s\nSTD=%s\n", mean, std)
            # keep the correct order, better interpretation
            self.normalize = StandardNormalize(mean, std, inplace=False)
            self._hf_backbone = hf_backbone

    @property
    def output_channels(self):
        return [self.config.num_channels] + self._hf_backbone.channels

    @property
    def output_strides(self):
        return [1] + self._output_strides

    def forward(self, x, *args, **kwargs):
        x = self.normalize(x)
        output = self._hf_backbone(x, *args, **kwargs)
        return [x] + output.feature_maps
