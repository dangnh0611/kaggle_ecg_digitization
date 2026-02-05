import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from yagm.layers.activation import get_act
from yagm.layers.norms import get_norm
from typing import Tuple, List


def init_neck_xavier(m):
    """Initialize weights for Conv3d layers using Xavier initialization."""
    if isinstance(m, nn.Conv3d):
        # Apply Xavier uniform initialization to Conv3d weights
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose3d):
        # Apply Xavier uniform initialization to ConvTranspose3d weights (if used)
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


# @TODO - support Normalization Layer
# @TODO - proper weight init ?
class FPN3d(nn.Module):
    def __init__(
        self,
        in_channels_list,
        out_channels_list=None,
        intermediate_channels_list=None,
        fusion_method="add",
        interpolation_method="trilinear",
        act="identity",
        norm="identity",
    ):
        super(FPN3d, self).__init__()
        self.interpolation_method = interpolation_method
        if out_channels_list is None:
            out_channels_list = in_channels_list
        if intermediate_channels_list is None:
            intermediate_channels_list = out_channels_list[:-1]

        num_levels = len(in_channels_list)
        assert (
            len(out_channels_list) == len(intermediate_channels_list) + 1 == num_levels
        )
        self._output_channels = out_channels_list

        if isinstance(fusion_method, str):
            self.fusion_methods = [fusion_method] * (num_levels - 1)
        else:
            assert len(fusion_method) == num_levels - 1
            self.fusion_methods = fusion_method

        # map feature map to (smaller) channels
        self.lateral_convs = nn.ModuleList()
        # map feature map to same channels to lateral connection, used in `add` only
        self.adapt_convs = nn.ModuleList()
        # smoothing convolution to merge and reduce artifact due to upsampling
        self.output_convs = nn.ModuleList()

        # bottom-up or coarse-to-fine
        for i in range(num_levels):
            if i == num_levels - 1:
                self.lateral_convs.append(
                    nn.Conv3d(
                        in_channels_list[i],
                        out_channels_list[i],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
                continue

            # Lateral convolution
            self.lateral_convs.append(
                nn.Conv3d(
                    in_channels_list[i],
                    intermediate_channels_list[i],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

            # Adapt convolution (only needed for levels < num_levels-1)
            if (
                self.fusion_methods[i] == "add"
                and out_channels_list[i + 1] != intermediate_channels_list[i]
            ):
                self.adapt_convs.append(
                    nn.Conv3d(
                        out_channels_list[i + 1],
                        intermediate_channels_list[i],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            else:
                self.adapt_convs.append(nn.Identity())

            # Output convolution
            if self.fusion_methods[i] == "concat":
                in_ch = intermediate_channels_list[i] + out_channels_list[i + 1]
            else:
                in_ch = intermediate_channels_list[i]
            output_conv = nn.Sequential(
                nn.Conv3d(
                    in_ch, out_channels_list[i], kernel_size=3, stride=1, padding=1
                ),
                get_act(act)(),
                get_norm(norm, channel_first=True)(out_channels_list[i]),
            )
            self.output_convs.append(output_conv)
        self.apply(init_neck_xavier)

    @property
    def output_channels(self):
        return self._output_channels

    def forward(self, inputs):
        laterals = [
            lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)
        ]

        output_features = [laterals[-1]]
        for i in range(len(laterals) - 2, -1, -1):
            higher_feature = output_features[-1]
            if self.fusion_methods[i] == "add":
                higher_feature = self.adapt_convs[i](higher_feature)

            upsampled_feature = F.interpolate(
                higher_feature,
                size=laterals[i].shape[2:],
                mode=self.interpolation_method,
                align_corners=False,
            )

            if self.fusion_methods[i] == "add":
                fused = laterals[i] + upsampled_feature
            else:
                fused = torch.cat([laterals[i], upsampled_feature], dim=1)
            out = self.output_convs[i](fused)
            output_features.append(out)
        return output_features[::-1]


class PAN3d(nn.Module):
    def __init__(
        self,
        in_channels_list,
        out_channels_list=None,
        intermediate_channels_list=None,
        fusion_method="add",
        interpolation_method="trilinear",
        act="identity",
        norm="identity",
    ):
        super().__init__()
        if out_channels_list is None:
            out_channels_list = in_channels_list
        if intermediate_channels_list is None:
            intermediate_channels_list = out_channels_list[:-1]
        self._output_channels = out_channels_list
        self.top_down = FPN3d(
            in_channels_list,
            out_channels_list,
            intermediate_channels_list,
            fusion_method,
            interpolation_method,
            act,
            norm,
        )
        self.bottom_up = FPN3d(
            out_channels_list[::-1],
            out_channels_list[::-1],
            intermediate_channels_list[::-1],
            fusion_method,
            interpolation_method,
            act,
            norm,
        )
        self.apply(init_neck_xavier)

    @property
    def output_channels(self):
        return self.bottom_up.output_channels[::-1]

    def forward(self, inputs):
        top_down_features = self.top_down(inputs)
        bottom_up_features = self.bottom_up(top_down_features[::-1])
        return bottom_up_features[::-1]


class FactorizedFPN3d(nn.Module):
    """
    Simpler version of FPN which fuse multi-level features map into one target level, e.g fusing C2, C3, C5 into C4
    Ref 2D implementation: https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/430491
    """

    def __init__(
        self,
        in_channels_list: list,
        intermediate_channels_list: list | int | None = None,
        target_level: int = -1,
        conv_ksizes=(3, 3),
        norm="layernorm_3d",
        act="gelu",
        interpolation_mode="trilinear",
    ):
        super().__init__()
        in_channels_list = list(in_channels_list)
        if intermediate_channels_list is None:
            intermediate_channels_list = in_channels_list[:]
        elif isinstance(intermediate_channels_list, (list, tuple)):
            assert len(intermediate_channels_list) == len(in_channels_list)
        elif isinstance(intermediate_channels_list, int):
            intermediate_channels_list = [intermediate_channels_list] * len(
                in_channels_list
            )
        else:
            raise ValueError

        self.num_levels = len(in_channels_list)
        # handle negative indexing
        self.target_level = (
            self.num_levels + target_level if target_level < 0 else target_level
        )
        self.interpolation_mode = interpolation_mode
        self._output_channels = in_channels_list[:]  # clone
        # keep non-target level only
        _target_level_in_channels = in_channels_list.pop(target_level)
        intermediate_channels_list.pop(target_level)

        # @TODO - bug here, keep for compatible only
        # if GLU, not GELU
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        in_ch,
                        mid_ch * 2,
                        kernel_size=conv_ksizes[0],
                        padding=1,
                    ),
                    get_act(act)(),
                    get_norm(norm, channel_first=True)(mid_ch * 2),
                    nn.Conv3d(
                        mid_ch * 2,
                        mid_ch,
                        kernel_size=conv_ksizes[1],
                        padding=1,
                    ),
                )
                for in_ch, mid_ch in zip(in_channels_list, intermediate_channels_list)
            ]
        )

        self._output_channels[target_level] += sum(intermediate_channels_list)

    @property
    def output_channels(self):
        return self._output_channels

    def forward(self, xs: list):
        assert len(xs) == self.num_levels
        target_feature = xs.pop(self.target_level)
        N, C, D, H, W = target_feature.shape
        intermediates = [target_feature]
        for i, (c, x) in enumerate(zip(self.convs, xs)):
            intermediates.append(
                F.interpolate(
                    c(x),
                    size=(D, H, W),
                    mode=self.interpolation_mode,
                    align_corners=None,
                )
            )
        target_feature = torch.cat(intermediates, dim=1)
        xs.insert(self.target_level, target_feature)
        return xs


class FactorizedFPN2d(nn.Module):
    """
    Simpler version of FPN which fuse multi-level features map into one target level, e.g fusing C2, C3, C5 into C4
    Ref 2D implementation: https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/430491
    """

    def __init__(
        self,
        in_channels_list: list,
        intermediate_channels_list: list | int | None = None,
        target_level: int = -1,
        conv_ksizes=(3, 3),
        norm="layernorm_2d",
        act="gelu",
        interpolation_mode="bilinear",
    ):
        super().__init__()
        in_channels_list = list(in_channels_list)
        if intermediate_channels_list is None:
            intermediate_channels_list = in_channels_list[:]
        elif isinstance(intermediate_channels_list, (list, tuple)):
            assert len(intermediate_channels_list) == len(in_channels_list)
        elif isinstance(intermediate_channels_list, int):
            intermediate_channels_list = [intermediate_channels_list] * len(
                in_channels_list
            )
        else:
            raise ValueError

        self.num_levels = len(in_channels_list)
        # handle negative indexing
        self.target_level = (
            self.num_levels + target_level if target_level < 0 else target_level
        )
        self.interpolation_mode = interpolation_mode
        self._output_channels = in_channels_list[:]  # clone
        # keep non-target level only
        _target_level_in_channels = in_channels_list.pop(target_level)
        intermediate_channels_list.pop(target_level)

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_ch,
                        mid_ch * 2 if act == "glu" else mid_ch,
                        kernel_size=conv_ksizes[0],
                        padding=1,
                    ),
                    get_act(act)(),
                    get_norm(norm, channel_first=True)(
                        mid_ch * 2 if act == "glu" else mid_ch
                    ),
                    nn.Conv2d(
                        mid_ch * 2 if act == "glu" else mid_ch,
                        mid_ch,
                        kernel_size=conv_ksizes[1],
                        padding=1,
                    ),
                )
                for in_ch, mid_ch in zip(in_channels_list, intermediate_channels_list)
            ]
        )

        self._output_channels[target_level] += sum(intermediate_channels_list)

    @property
    def output_channels(self):
        return self._output_channels

    def forward(self, xs: list):
        assert len(xs) == self.num_levels
        target_feature = xs.pop(self.target_level)
        N, C, H, W = target_feature.shape
        intermediates = [target_feature]
        for i, (c, x) in enumerate(zip(self.convs, xs)):
            intermediates.append(
                F.interpolate(
                    c(x),
                    size=(H, W),
                    mode=self.interpolation_mode,
                    align_corners=None,
                )
            )
        target_feature = torch.cat(intermediates, dim=1)
        # print('NECK:', target_feature.shape, [e.shape for e in intermediates])
        xs.insert(self.target_level, target_feature)
        return xs


if __name__ == "__main__":
    import torch

    BS = 3
    INPUT_SIZE = (128, 224, 224)
    D, H, W = INPUT_SIZE
    STRIDES = (2, 4, 16, 32)
    CHANNELS = [128, 256, 512, 1024]
    # OUT_CHANNELS = [256, 256, 256, 256]
    OUT_CHANNELS = [122, 133, 155, 166]
    FEATURE_SHAPES = [(D // stride, H // stride, W // stride) for stride in STRIDES]
    print("FEATURES SHAPE:\n", FEATURE_SHAPES)
    inps = [
        torch.rand((BS, C, *_spatial_shape), dtype=torch.float32)
        for C, _spatial_shape in zip(CHANNELS, FEATURE_SHAPES)
    ]

    # model = FPN3d(
    #     CHANNELS,
    #     out_channels_list=OUT_CHANNELS,
    #     fusion_method="add",
    #     interpolation_method="trilinear",
    # )

    # model = PAN3d(
    #     CHANNELS,
    #     out_channels_list=OUT_CHANNELS,
    #     fusion_method="add",
    #     interpolation_method="trilinear",
    #     act = 'relu',
    #     norm = 'layernorm_3d'
    # )

    model = FactorizedFPN3d(
        CHANNELS,
        128,
        target_level=-2,
        conv_ksizes=(3, 3),
        norm="layernorm_3d",
        act="relu",
        interpolation_mode="trilinear",
    )

    model = model.eval()
    print(model)

    outs = model(inps)
    print("OUTPUTS:", [e.shape for e in outs])
