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
from yagm.layers.common import MLP, ConvNormAct2d, SegmentationHead2d
from yagm.layers.norms import get_norm

logger = logging.getLogger(__name__)


class ColumnDSNT(nn.Module):
    """
    Differentiable spatial to numerical transform (DSNT)
    Ref: https://arxiv.org/abs/1801.07372
    Modified version worked with column-wise Softmax
    """

    def __init__(self, height, width, act):
        super().__init__()
        self.act = act
        self._need_col_norm = not isinstance(act, nn.Softmax)
        # (H,) -> (1,1,H,W) in xy order
        self.grid = torch.linspace(0.5 / height, 1 - 0.5 / height, height)[
            None, None, :, None
        ].repeat(1, 1, 1, width)

    def forward(self, heatmap):
        """
        Args:
            heatmap: heatmap raw logits of shape (N, C, H, W)
        Returns:
            Normalized coordinates (x,y) in range [0,1] in contigous coordinate space
            shape (N, C, W)
        """
        B, C, H, W = heatmap.shape
        heatmap = self.act(heatmap)
        if self._need_col_norm:
            # normalize so that sum over column dims equals to 1.0
            prob = heatmap / heatmap.sum(dim=2, keepdim=True)
        else:
            # already a probability distribution
            prob = heatmap
        # (1,1,H,W) * (N,C,H,W) -> (N,C,H,W) -> (N,C,W)
        y = torch.sum(self.grid.to(prob) * prob, dim=2)
        return y


class SignalHeatmapModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self.cfg = cfg
        loss_weights = global_cfg.loss.mtl.loss_weights
        # heatmap_loss, offset_loss, combined_loss
        assert len(loss_weights) == 3
        # assert loss_weights[0] > 0
        self.subpixel_mode = cfg.subpixel_mode

        # ========== ENCODER ==========
        self.encoder: BackboneMixin = hydra.utils.instantiate(cfg.encoder)

        # this include the input channels (stride 1) as 1st element
        enc_channels = self.encoder.output_channels
        enc_strides = self.encoder.output_strides
        assert len(enc_channels) == len(enc_strides)

        # -----MONKEY-PATCHING-----
        # change the stem stride
        if cfg.change_stem_stride is not None:
            new_stride = int(cfg.change_stem_stride)
            if "convnext" in cfg.encoder.model_name:
                # this work for ConvNext
                stem_conv = self.encoder._backbone.stem[0]
                stem_conv.stride = (new_stride, new_stride)
                stem_conv.padding = "same" if new_stride % 2 == 1 else 0
            elif "coat" in cfg.encoder.model_name:
                # this work for Coat
                stem_conv = self.encoder._backbone.patch_embed1.proj
                stem_conv.stride = (new_stride, new_stride)
                stem_conv.padding = "same" if new_stride % 2 == 1 else 0
            else:
                raise ValueError
            stride_scale_factor = enc_strides[1] // new_stride
            old_enc_strides = enc_strides
            enc_strides = [
                e // stride_scale_factor if i != 0 else e
                for i, e in enumerate(enc_strides)
            ]
            logger.info(
                "Change encoder strides from %s to %s", old_enc_strides, enc_strides
            )

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

        heatmap_loss_name = global_cfg.loss.heatmap_loss
        if heatmap_loss_name == "bce":
            self.heatmap_act = nn.Sigmoid()
        elif heatmap_loss_name in ["l1", "mse"]:
            self.heatmap_act = nn.Identity()
        elif heatmap_loss_name in ["cce", "kl", "bikl", "jsd"]:
            # column-wise softmax
            self.heatmap_act = nn.Softmax(dim=2)
        else:
            raise ValueError

        # DSNT heatmap to coordinate
        if self.subpixel_mode == "dsnt":
            self.col_dsnt = ColumnDSNT(
                global_cfg.data.heatmap_hwl[0],
                global_cfg.data.heatmap_hwl[1],
                self.heatmap_act,
            )

        # ========== WEIGHT INITIALIZATION ==========
        # @TODO - proper init the decoder

        # init segmentation head
        # 1e-4 just a dirty approximation, never mind
        pos_rates = torch.tensor(5e-3)
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

        # Heatmap
        heatmap = self.seg_head(dec_features[0])  # finest resolution

        # DSNT
        if self.subpixel_mode is None:
            assert heatmap.shape[1] == 1
            prob_heatmap = heatmap
            offset_heatmap = None
            combined_signal = None
        elif self.subpixel_mode == "dsnt":
            assert heatmap.shape[1] == 1
            prob_heatmap = heatmap
            offset_heatmap = None
            combined_signal = self.col_dsnt(prob_heatmap)
        elif self.subpixel_mode == "offset":
            raise NotImplementedError
            assert heatmap.shape[0] == 2
            offset_heatmap = heatmap[1:]  # shape (1, H, W)
            combined_signal = None
        else:
            raise ValueError

        return prob_heatmap, offset_heatmap, combined_signal
