# @TODO: add norm before Unet decoder ?

import logging
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import hydra
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from timm.data import resolve_data_config
from timm.layers import DropPath, LayerNorm2d
from timm.models._features import feature_take_indices
from yagm.backbones import BackboneMixin
from yagm.layers.activation import get_act
from yagm.layers.common import MLP, ConvNormAct2d, DSNT2d, SegmentationHead2d
from yagm.layers.norms import get_norm

logger = logging.getLogger(__name__)


class SpatialAwareRegressionHead2d(nn.Module):

    def __init__(
        self,
        feature_h: int,
        feature_w: int,
        in_features: int,
        num_outputs: int,
        channels_per_output: int,
        proj_norm: Callable,
        proj_channels_per_output: int = 32,
        proj_drop_rate: float = 0.0,
        hidden_channels=tuple(),
        drop_rate: float = 0.0,
        norm_layer: Callable = None,
        act_layer: Callable = None,
    ):
        """
        No pooling, flatten along spatial dimensions
        """
        super().__init__()
        self.K = num_outputs
        self.D = proj_channels_per_output
        self.H = feature_h
        self.W = feature_w

        self.proj_norm = proj_norm(in_features)
        self.proj_conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=self.K * self.D,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_output_channels = self.D * self.H * self.W
        self.proj_act = act_layer()
        self.proj_drop = nn.Dropout(proj_drop_rate)

        if drop_rate > 0.0:
            logger.warning("Dropout was set to %f > 0 in regression head!", drop_rate)
        self.reg_head = MLP(
            in_channels=self.proj_output_channels,
            hidden_channels=hidden_channels + [channels_per_output],
            norm_layer=norm_layer,
            act_layer=act_layer,
            bias=True,
            dropout=drop_rate,
            last_norm=False,
            last_activation=False,
            last_dropout=False,
        )

    def forward(self, x):
        x = self.proj_drop(self.proj_act(self.proj_conv(self.proj_norm(x))))
        B, KD, H, W = x.shape
        x = x.reshape(B, self.K, self.D, H, W).reshape(B, self.K, self.D * H * W)
        reg_logits = self.reg_head(x)  # (B, K, channels_per_output)
        return reg_logits


class KeypointDetectionModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self.cfg = cfg
        loss_weights = global_cfg.loss.mtl.loss_weights
        assert len(loss_weights) == 4
        # assert loss_weights[0] > 0
        self._enable_dsnt_head = loss_weights[1] > 0
        self._enable_sa_reg_head = loss_weights[2] > 0

        # ========== ENCODER ==========
        self.encoder: BackboneMixin = hydra.utils.instantiate(cfg.encoder)

        # this include the input channels (stride 1) as 1st element
        enc_channels = self.encoder.output_channels
        enc_strides = self.encoder.output_strides
        assert len(enc_channels) == len(enc_strides)

        # -----MONKEY-PATCHING-----
        # change the stem stride
        if cfg.change_stem_stride is not None:
            # this work for ConvNext
            stem_conv = self.encoder.stem[0]
            stem_conv.stride = (
                int(cfg.change_stem_stride),
                int(cfg.change_stem_stride),
            )
            stem_conv.padding = 'same'
            stride_scale_factor = enc_strides[1] // int(cfg.change_stem_stride)
            old_enc_strides = enc_strides
            enc_strides = [
                e // stride_scale_factor if i != 0 else e
                for i, e in enumerate(enc_strides)
            ]
            logger.info('Change encoder strides from %s to %s', old_enc_strides, enc_strides)

        logger.info(
            "Encoder multiscale feature has channels=%s strides=%s",
            enc_channels,
            enc_strides,
        )

        # ========== NECK ==========
        if getattr(cfg, "neck", None) is not None:
            neck_kwargs = OmegaConf.to_object(cfg.neck)
            neck_cls = hydra.utils.get_object(neck_kwargs.pop("_target_"))
            # ignore the first (stride 1) feature map
            self.neck = neck_cls(enc_channels[1:], **neck_kwargs)
        else:
            self.neck = None
        print("Before Neck feature channels:", enc_channels)
        if self.neck is not None:
            enc_channels = enc_channels[0:1] + self.neck.output_channels
        print("After Neck feature channels:", enc_channels)

        if cfg.freeze_encoder:
            logger.info("Freeze weights of encoder")
            for param in self.encoder.parameters():
                param.requires_grad = False
            # important for BatchNorm
            self.encoder.eval()

        # ========== UNET DECODER ==========
        decoder_kwargs = OmegaConf.to_object(cfg.decoder)
        decoder_cls = hydra.utils.get_object(decoder_kwargs.pop("_target_"))
        self.decoder = decoder_cls(
            encoder_channels=enc_channels,  # fine to coarse
            encoder_strides=enc_strides,  # fine to coarse)
            **decoder_kwargs,
        )
        dec_channels = self.decoder.output_channels

        # ========== SEGMENTATION HEAD ==========
        self.seg_head = SegmentationHead2d(
            in_channels=dec_channels[0],
            out_channels=cfg.seg_head.out_channels,
            ksize=cfg.seg_head.ksize,
            act=None,
            hidden_channels=cfg.seg_head.hidden_channels,
            hidden_norm=cfg.seg_head.hidden_norm,
            hidden_act=cfg.seg_head.hidden_act,
        )

        self.enc_channels = enc_channels
        self.enc_strides = enc_strides
        self.dec_channels = dec_channels

        # ========== Spatial Aware (SA) regression head ==========
        # SimmCC like, but direct regression instead of bin-based regression by classification
        if self._enable_sa_reg_head:
            self.sa_reg_feature_mode = cfg.sa_reg_head.feature_mode
            self.sa_reg_feature_idx = cfg.sa_reg_head.feature_idx
            sa_reg_feature_channel, sa_reg_feature_stride = self.get_feature_channels(
                self.sa_reg_feature_mode, self.sa_reg_feature_idx
            )

            self.sa_reg_head = SpatialAwareRegressionHead2d(
                feature_h=global_cfg.data.img_size[0] // sa_reg_feature_stride,
                feature_w=global_cfg.data.img_size[1] // sa_reg_feature_stride,
                in_features=sa_reg_feature_channel,
                num_outputs=57,
                channels_per_output=2,
                proj_norm=get_norm(
                    f"{cfg.sa_reg_head.norm}_2d",
                    channel_first=True,
                    eps=cfg.sa_reg_head.norm_eps,
                ),
                proj_channels_per_output=cfg.sa_reg_head.proj_channels_per_output,
                proj_drop_rate=cfg.sa_reg_head.proj_drop_rate,
                hidden_channels=cfg.sa_reg_head.hidden_channels,
                drop_rate=cfg.sa_reg_head.drop_rate,
                norm_layer=get_norm(
                    cfg.sa_reg_head.norm,
                    channel_first=False,
                    eps=cfg.sa_reg_head.norm_eps,
                ),
                act_layer=get_act(cfg.sa_reg_head.act, inplace=True),
            )

        # DSNT heatmap to coordinate
        if self._enable_dsnt_head:
            heatmap_loss_name = global_cfg.loss.main.heatmap_loss
            if heatmap_loss_name == "bce":
                _heatmap_act = "sigmoid"
            elif heatmap_loss_name in ["l1", "mse"]:
                _heatmap_act = "identity"
            elif heatmap_loss_name in ["kl", "bikl", "jsd"]:
                _heatmap_act = "softmax"
            else:
                raise ValueError

            self.dsnt = DSNT2d(
                global_cfg.data.img_size[0] // global_cfg.data.heatmap_stride[0],
                global_cfg.data.img_size[1] // global_cfg.data.heatmap_stride[1],
                act=_heatmap_act,
            )

        # ========== WEIGHT INITIALIZATION ==========
        # @TODO - proper init the decoder

        # init segmentation head
        # 1e-4 just a dirty approximation, never mind
        pos_rates = torch.tensor(1e-4)
        bias_init = -torch.log(1.0 / pos_rates - 1.0)
        nn.init.constant_(self.seg_head[-1].bias, bias_init)
        logger.info("Init bias value of segmentation head to %s", bias_init)

    def get_feature_channels(self, feature_mode, feature_idx):
        assert feature_idx < 0, "Currently support negative indexing only"
        assert feature_mode in ["enc", "dec", "enc_dec"]
        if feature_mode == "enc":
            channels_list = self.enc_channels
        elif feature_mode == "dec":
            channels_list = self.dec_channels  # fine to coarse
        elif feature_mode == "enc_dec":
            channels_list = []
            # reverse so that current order is all coarse to fine
            # all start from the coarsest feature map (usually stride 32)
            _enc_channels = enc_channels[::-1]
            _dec_channels = dec_channels[::-1]
            for i in range(max(len(_enc_channels), len(_dec_channels))):
                if i == 0:
                    # usually, center_block=False in decoder
                    # @TODO - handle other cases
                    assert _enc_channels[0] == _dec_channels[0]
                    block_channel = _dec_channels[0]
                else:
                    _from_enc = _enc_channels[i] if i < len(_enc_channels) else 0
                    _from_dec = _dec_channels[i] if i < len(_dec_channels) else 0
                    block_channel = _from_enc + _from_dec
                channels_list.append(block_channel)
            # fine to coarse order
            channels_list = channels_list[::-1]
        else:
            raise ValueError

        select_feature_stride = self.enc_strides[feature_idx]
        select_feature_channel = channels_list[feature_idx]
        logger.info(
            "\nEncoder channels:%s\n Decoder channels:%s\n Channels list:%s\n Select feature channel:%d\n Select feature stride:%d",
            self.enc_channels,
            self.dec_channels,
            channels_list,
            select_feature_channel,
            select_feature_stride,
        )
        return select_feature_channel, select_feature_stride

    def get_feature(self, enc_features, dec_features, feature_mode, feature_idx):
        if feature_idx == -1:
            feature = enc_features[-1]
        elif feature_mode == "enc":
            feature = enc_features[feature_idx]
        elif feature_mode == "dec":
            feature = dec_features[feature_idx]
        elif feature_mode == "enc_dec":
            feature = torch.cat(
                [
                    enc_features[feature_idx],
                    dec_features[feature_idx],
                ],
                dim=1,
            )
        else:
            raise ValueError
        return feature

    def forward(self, x):
        enc_features = self.encoder(x)
        # print('encoder features:', [tuple(e.shape) for e in enc_features])

        if self.neck is not None:
            enc_features = enc_features[0:1] + self.neck(enc_features[1:])

        dec_features = self.decoder(enc_features)

        # SA Regression
        if self._enable_sa_reg_head:
            sa_reg_feature = self.get_feature(
                enc_features,
                dec_features,
                self.sa_reg_feature_mode,
                self.sa_reg_feature_idx,
            )
            sa_reg_logits = self.sa_reg_head(sa_reg_feature)
        else:
            sa_reg_logits = None

        # Heatmap
        heatmap = self.seg_head(dec_features[0])  # finest resolution

        # DSNT
        if self._enable_dsnt_head:
            # first channel is grid heatmap prediction
            dsnt_kpt = self.dsnt(heatmap[:, 1:])
        else:
            dsnt_kpt = None

        # print(heatmap.shape, dsnt_kpt.shape, sa_reg_logits.shape)
        return heatmap, dsnt_kpt, sa_reg_logits
