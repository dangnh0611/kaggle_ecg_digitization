from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from yagm.layers.activation import get_act
from yagm.layers.norms import get_norm
from timm.layers import LayerNorm2d

__all__ = [
    "TransposeLast",
    "MaskedSoftmax1d",
    "MLP",
    "SameConv1d",
    "MaskedConv1d",
    "CausalConv1d",
    "ConvNormAct1d",
    "ConvActNorm1d",
    "ConvNormAct2d",
    "ConvActNorm2d",
    "make_divisible",
    "make_net_dims",
]


class TransposeLast(nn.Module):

    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)


class MaskedSoftmax1d(nn.Module):

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, L)
            mask: (N, L) where where pad=False, nonpad=True
        Returns:
            (N, L)
        """
        if mask is not None:
            x = x.masked_fill(~mask, torch.finfo(x.dtype).min)
        return F.softmax(x, dim=self.dim)


class MLP(torch.nn.Sequential):
    """Modify from torchvision's MLP, add option for last Norm/Activation/Dropout.
    This block implements the multi-layer perceptron (MLP) module.
    Args:
        in_channels: Number of channels of the input
        hidden_channels: List of the hidden channel dimensions
        norm_layer: Norm layer that will be stacked on top of the linear layer.
            If ``None`` this layer won't be used. Default: ``None``
        activation_layer: Activation function which will be stacked on top of the normalization layer
            (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used.
            Default: `torch.nn.ReLU`
        inplace: Parameter for the activation layer, which can optionally do the operation in-place.
            Default is `None`, which uses the respective default values of the `activation_layer` and Dropout layer.
        bias: Whether to use bias in the linear layer. Default ``True``
        dropout: The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        act_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
        last_norm=True,
        last_activation=True,
        last_dropout=True,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        layers = []
        in_dim = in_channels
        for i, hidden_dim in enumerate(hidden_channels):
            is_not_last = i != len(hidden_channels) - 1
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None and (is_not_last or last_norm):
                layers.append(norm_layer(hidden_dim))
            if is_not_last or last_activation:
                layers.append(act_layer())
            if is_not_last or last_dropout:
                layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        super().__init__(*layers)


class SameConv1d(nn.Conv1d):
    """Convolution 1D layer without changing sequence length L.
    Input: (N, C1, L)
    Output: (N, C2, L), EXACTLY same sequence length as Input
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        if dilation != 1:
            raise ValueError("dilation != 1 is currently not supported!")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def _calc_same_pad(self, seq_len, ksize, stride):
        if seq_len % stride == 0:
            pad = max(ksize - stride, 0)
        else:
            pad = max(ksize - (seq_len % stride), 0)
        return pad

    def forward(self, x):
        pad = self._calc_same_pad(x.size(-1), self.kernel_size[0], self.stride[0])
        x = F.pad(x, [pad // 2, pad - pad // 2])
        return super().forward(x)


class SameDepthwiseConv1d(SameConv1d):
    """Depthwise Convolution 1D layer without changing sequence length L.
    Input: (N, C1, L)
    Output: (N, C2, L), EXACTLY same sequence length as Input
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=True,
    ):
        assert out_channels % in_channels == 0
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )


class MaskedConv1d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=17,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = SameConv1d(
            in_channels,
            out_channels,
            kernel_size,
            groups=groups,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.supports_masking = True

    def _compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.stride > 1:
                mask = mask[:, :: self.stride]
        return mask

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(
                ~mask[:, None, :], torch.tensor(0.0, dtype=x.dtype, device=x.device)
            )
            mask = self._compute_mask(x, mask)
        x = self.conv(x)
        return x, mask


class CausalConv1d(nn.Sequential):
    """Causal Convolution 1D, where token at index i is only mixed with previous token 0..i-1
    This ensures CAUSALITY and sometime reduce the effect of padding (train-test gap) in CNN 1D model.
    Ref: https://www.kaggle.com/competitions/asl-signs/discussion/406684
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CausalConv1d, self).__init__(
            nn.ConstantPad1d((dilation * (kernel_size - 1), 0), 0.0),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding="valid",
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )


class ConvNormAct1d(nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        act_layer=partial(nn.ReLU, inplace=True),
        norm_layer=partial(nn.BatchNorm1d),
    ):
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=(norm_layer is None),
        )
        if norm_layer is None:
            norm = nn.Identity()
        else:
            norm = norm_layer(out_channels)
        act = act_layer()
        super().__init__(conv, norm, act)


class ConvActNorm1d(nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        act_layer=partial(nn.ReLU, inplace=True),
        norm_layer=partial(nn.BatchNorm1d),
    ):
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=(norm_layer is None),
        )
        act = act_layer()
        if norm_layer is None:
            norm = nn.Identity()
        else:
            norm = norm_layer(out_channels)
        super().__init__(conv, act, norm)


class ConvNormAct2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
        conv_layer=nn.Conv2d,
    ):
        super().__init__()
        # ---- conv2d ----
        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            )
        ]
        # exit()
        # ---- normalization ----
        if norm_layer is not None:
            if isinstance(norm_layer, str):
                norm_layer = get_norm(f"{norm_layer}_2d", channel_first=True)
            layers.append(norm_layer(out_channels))

        # ---- activation ----
        if act_layer is not None:
            if isinstance(act_layer, str):
                act_layer = get_act(act_layer, inplace=True)
            layers.append(act_layer())

        super().__init__(*layers)


class ConvActNorm2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
        conv_layer=nn.Conv2d,
    ):
        """
        A convenient block: Conv2d → Normalization → Activation.

        Args:
            in_channels:  Number of input channels.
            out_channels: Number of output channels.
            kernel_size:  Size of the convolution kernel.
            stride:       Convolution stride.
            padding:      Padding (if None, uses 'same' style for kernel_size).
            dilation:     Dilation rate.
            groups:       Groups in Conv2d.
            bias:         Bias for Conv2d (usually False with BatchNorm).
            norm_layer:   Normalization layer class (e.g. nn.BatchNorm2d, nn.GroupNorm, None).
            activation_layer: Activation function instance (e.g. nn.ReLU(), nn.LeakyReLU(), None).
        """
        super().__init__()

        # Auto padding like 'same' if not given
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                padding_mode,
            )
        ]

        # Activation (optional)
        if act_layer is not None:
            layers.append(act_layer())

        # Normalization (optional)
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        super().__init__(*layers)


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def make_net_dims(depth, base_dim, dim_scale_method, width_multiplier=1, divisor=8):
    ret_dims = [base_dim]
    if "add" in dim_scale_method:
        _method, coef = dim_scale_method.split("_")
        assert _method == "add"
        coef = float(coef)
        for _ in range(depth):
            ret_dims.append(make_divisible(ret_dims[-1] + coef, divisor))
    elif "mul" in dim_scale_method:
        _method, coef = dim_scale_method.split("_")
        assert _method == "mul"
        coef = float(coef)
        for _ in range(depth):
            ret_dims.append(make_divisible(int(ret_dims[-1] * coef), divisor))
    else:
        raise ValueError
    ret_dims = [make_divisible(e * width_multiplier, divisor) for e in ret_dims]
    return ret_dims[1:]


class LSTMBlock2d(nn.Module):

    def __init__(self, in_chs, bidirectional=False, num_layers=1):
        """2D LSTM to encode a sequence of 2D images along time/depth dimension.
        Expect input of shape (B, T, C, H, W) or (B, D, C, H, W)
        Args:
            in_chs: number of input channels
            bidirectional: use bidirectional LSTM or not
            num_layers: number of LSTM layers
        """
        super().__init__()
        self.lstm = nn.LSTM(
            in_chs,
            in_chs if not bidirectional else in_chs // 2,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )

    def forward(self, x):
        # @TODO - optimize?
        B, T, C, H, W = x.shape
        # BTCHW -> BTC(HW) -> B(HW)TC -> (BHW)TC
        x = x.flatten(-2, -1).permute(0, 3, 1, 2).flatten(0, 1)
        x = self.lstm(x)[0]  # (BHW)TC
        x = x.view(-1, H, W, T, C).permute(0, 3, 4, 1, 2)  # BHWTC -> BTCHW
        return x


class TimeToChannelSlidingResample(nn.Module):

    def __init__(self, stride=1, channels=3, padding_mode="replicate"):
        """
        Sliding Resample along temporal/depth dimension,
        effectively convert time/depth to channels dimension.
        Useful for 2.5D image/voxel/video modeling
            Input shape: (B, T1, ...)
            Ouput shape: (B, T2, C, ...)
        """
        super().__init__()
        assert stride >= 1 and stride <= channels
        self.stride = stride
        self.channels = channels
        self.padding_mode = padding_mode

    def forward(self, x):
        """
        Args:
            x: (B, T, ...)
        """
        T = x.size(1)
        if self.channels > 2:
            # right padding
            pad = [0] * ((len(x.shape) - 2) * 2) + [
                (self.channels - 1) // 2,
                (self.channels - 1) // 2,
            ]
            x = F.pad(x, pad=pad, mode=self.padding_mode)
        xs = [
            x[:, offset : T + offset : self.stride] for offset in range(self.channels)
        ]  # List of tensor with shape (B, T2, ...)
        x = torch.stack(xs, dim=2)  # (B, T2, C, ...)
        return x


class StandardNormalize(nn.Module):

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        if self.inplace:
            x.sub_(self.mean).div_(self.std)
        else:
            return (x - self.mean) / self.std


# @TODO - also check with torch.nn.Upsample
class Interpolate(nn.Module):
    """
    Module wrapper for torch.nn.functional.interpolate that supports static or dynamic size.

    Parameters printed in repr:
        size, scale_factor, mode, align_corners, recompute_scale_factor, antialias
    """

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
        antialias: bool = False,
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def forward(
        self,
        x: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> torch.Tensor:
        """
        F.interpolate forward pass.
        Runtime `size` / `scale_factor` override constructor values.
        """
        size = size if size is not None else self.size
        scale_factor = scale_factor if scale_factor is not None else self.scale_factor

        if size is None and scale_factor is None:
            return x  # passthrough

        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias,
        )

    # ------------------------------------------------------------------
    # PyTorch-style pretty printing
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        """
        Build a repr string similar to nn.Upsample, nn.Conv2d, etc.
        Only prints fields that differ from defaults or are meaningful.
        """

        args = []

        # size or scale_factor
        if self.size is not None:
            args.append(f"size={self.size}")
        if self.scale_factor is not None:
            args.append(f"scale_factor={self.scale_factor}")

        # standard params
        args.append(f"mode='{self.mode}'")

        ac = (
            "None"
            if self.align_corners is None
            else ("True" if self.align_corners else "False")
        )
        args.append(f"align_corners={ac}")

        rsf = (
            "None"
            if self.recompute_scale_factor is None
            else ("True" if self.recompute_scale_factor else "False")
        )
        args.append(f"recompute_scale_factor={rsf}")

        # antialias
        if self.antialias:
            args.append("antialias=True")

        return ", ".join(args)


class SegmentationHead2d(nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels: int,
        ksize: int = 3,
        act=None,
        hidden_channels: Tuple[int] | None = None,
        hidden_norm=LayerNorm2d,
        hidden_act=nn.GELU,
    ):
        # ---- normalization ----
        if hidden_norm is not None:
            if isinstance(hidden_norm, str):
                hidden_norm = get_norm(f"{hidden_norm}_2d")

        # ---- activation ----
        if hidden_act is not None:
            if isinstance(hidden_act, str):
                hidden_act = get_act(hidden_act, inplace=True)

        layers = []
        cur_ch = in_channels
        if hidden_channels is not None and len(hidden_channels) > 0:
            for hidden_ch in hidden_channels:
                # x2 MLP expand if not provided
                hidden_ch = hidden_ch if hidden_ch > 0 else cur_ch * 2
                layers.extend(
                    [
                        nn.Conv2d(cur_ch, hidden_ch, ksize, padding="same"),
                        hidden_norm(hidden_ch),
                        hidden_act(),
                    ]
                )
                cur_ch = hidden_ch
        # final convolution
        layers.append(nn.Conv2d(cur_ch, out_channels, ksize, padding="same"))
        if act is not None:
            layers.append(act())
        super().__init__(*layers)


class SpatialSoftmax2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.flatten(x, -2, -1)
        prob = F.softmax(x, dim=-1).reshape(B, C, H, W)
        return prob


class DSNT2d(nn.Module):
    """
    Differentiable spatial to numerical transform (DSNT)
    Ref: https://arxiv.org/abs/1801.07372
    """

    def __init__(self, height, width, act="sigmoid"):
        super().__init__()
        self._act = act
        # xy
        self.grid = self.create_normalized_meshgrid(height, width)
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "identity":
            self.act = nn.Identity()
        elif act == "softmax":
            self.act = SpatialSoftmax2d()
        else:
            raise ValueError

    def create_normalized_meshgrid(self, height, width):
        """
        Return meshgrid in range [0, 1], shape (H, W, 2)
        """
        xs = torch.linspace(0.5 / width, 1 - 0.5 / width, width)
        ys = torch.linspace(0.5 / height, 1 - 0.5 / height, height)
        grid = torch.stack(torch.meshgrid([xs, ys], indexing="xy"), dim=-1)
        return grid

    def forward(self, heatmap):
        """
        Args:
            heatmap: heatmap raw logits of shape (N, C, H, W)
        Returns:
            Normalized coordinates (x,y) in range [0,1] in coordinate space (not pixel space)
            shape (N, C, 2)
        """
        B, C, H, W = heatmap.shape
        heatmap = self.act(heatmap)
        if self._act != "softmax":
            # normalize so that sum over spatial dims equals to 1
            prob = heatmap / heatmap.sum(dim=(2, 3), keepdim=True)
        else:
            # already a probability distribution
            prob = heatmap
        
        # (1,1,H,W,2) * (N,C,H,W,1) -> (N,C,H,W,2) -> (N,C,2)
        xy = torch.sum(self.grid[None, None].to(prob) * prob[..., None], dim=(2, 3))
        return xy
