# @TODO: add norm before Unet decoder ?

import logging
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Union

import hydra
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from timm.layers import SelectAdaptivePool2d
from yagm.layers.activation import get_act
from yagm.layers.common import MLP
from yagm.layers.norms import get_norm

logger = logging.getLogger(__name__)


class ImageRotationModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self._model_name = cfg.encoder.model_name

        # ========== ENCODER ==========
        self.encoder: BackboneMixin = hydra.utils.instantiate(cfg.encoder)

        # this include the input channels (stride 1) as 1st element
        enc_channels = self.encoder.output_channels
        enc_strides = self.encoder.output_strides
        assert len(enc_channels) == len(enc_strides)

        # =========== HEADS ==========
        pooled_feature_channel = enc_channels[-1]
        if cfg.head.pool_type == "catavgmax":
            pooled_feature_channel *= 2

        self.global_pool = SelectAdaptivePool2d(
            pool_type=cfg.head.pool_type, flatten=True
        )
        self.dropout = nn.Dropout(p=cfg.head.dropout)

        head_norm_layer = (
            get_norm(cfg.head.norm, channel_first=True)
            if cfg.head.norm is not None
            else None
        )
        head_act_layer = get_act(cfg.head.act, inplace=True)
        self.cls_head = MLP(
            in_channels=pooled_feature_channel,
            hidden_channels=[*cfg.head.hidden_channels, 4],
            norm_layer=head_norm_layer,
            act_layer=head_act_layer,
            bias=True,
            dropout=cfg.head.hidden_dropout,
            last_norm=False,
            last_activation=False,
            last_dropout=False,
        )

        # don't use dropout on regression head
        self.reg_head = MLP(
            in_channels=pooled_feature_channel,
            hidden_channels=[*cfg.head.hidden_channels, 2],
            norm_layer=head_norm_layer,
            act_layer=head_act_layer,
            bias=True,
            dropout=0,
            last_norm=False,
            last_activation=False,
            last_dropout=False,
        )

    def forward(self, x):
        # latest/coarest feature map
        x = self.encoder(x)[-1]
        x = self.global_pool(x)
        x_drop = self.dropout(x)
        cls_logit = self.cls_head(x_drop)
        reg_logit = self.reg_head(x)
        return cls_logit, reg_logit
