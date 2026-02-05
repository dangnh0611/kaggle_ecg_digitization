import logging
import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import norm as timm_norm
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

logger = logging.getLogger(__name__)

__all__ = [
    "get_norm",
    "replace_norm_inplace",
    "MaskedBatchNorm1d",
    "MaskedBatchNorm1dV2",
    "LayerScale",
    "IBNorm3d",
    "IBNorm2d",
    "IBNorm1d",
    "GhostBatchNorm1d",
    "ChannelLastBatchNorm1d",
    "ChannelLastBatchNorm2d",
    "ChannelLastMaskedBatchNorm1d",
    "ChannelFirstLayerNorm1d",
    "ChannelFirstLayerNorm2d",
    "ChannelLastInstanceNorm1d",
]


def _masked_batch_norm(
    input: Tensor,
    mask: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float = 1e-5,
) -> Tensor:
    # from https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    r"""Applies Masked Batch Normalization for each channel in each data sample in a batch.
    See :class:`~MaskedBatchNorm1d`, :class:`~MaskedBatchNorm2d`, :class:`~MaskedBatchNorm3d` for details.
    """
    if not training and (running_mean is None or running_var is None):
        raise ValueError(
            "Expected running_mean and running_var to be not None when training=False"
        )

    num_dims = len(input.shape[2:])
    _dims = (0,) + tuple(range(-num_dims, 0))
    _slice = (None, ...) + (None,) * num_dims

    if training:
        num_elements = mask.sum(_dims)
        mean = (input * mask).sum(_dims) / num_elements  # (C,)
        var = (((input - mean[_slice]) * mask) ** 2).sum(_dims) / num_elements  # (C,)

        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) + momentum * mean.detach())
        if running_var is not None:
            running_var.copy_(running_var * (1 - momentum) + momentum * var.detach())
    else:
        mean, var = running_mean, running_var

    out = (input - mean[_slice]) / torch.sqrt(var[_slice] + eps)  # (N, C, ...)

    if weight is not None and bias is not None:
        out = out * weight[_slice] + bias[_slice]

    return out


class _MaskedBatchNorm(_BatchNorm):
    # from https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(_MaskedBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        self._check_input_dim(input)
        if mask is not None:
            self._check_input_dim(mask)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if mask is None:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                (
                    self.running_mean
                    if not self.training or self.track_running_stats
                    else None
                ),
                (
                    self.running_var
                    if not self.training or self.track_running_stats
                    else None
                ),
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            return _masked_batch_norm(
                input,
                mask,
                self.weight,
                self.bias,
                (
                    self.running_mean
                    if not self.training or self.track_running_stats
                    else None
                ),
                (
                    self.running_var
                    if not self.training or self.track_running_stats
                    else None
                ),
                bn_training,
                exponential_average_factor,
                self.eps,
            )


class MaskedBatchNorm1d(torch.nn.BatchNorm1d, _MaskedBatchNorm):
    # from https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    r"""Applies Batch Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..
    See documentation of :class:`~torch.nn.BatchNorm1d` for details.
    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        channels_last: bool = False,
    ) -> None:
        super(MaskedBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.channels_last = channels_last

    def forward(self, inputs, mask=None):
        if self.channels_last:
            inputs = inputs.permute(0, 2, 1)
        if mask is not None:
            mask = mask[:, None, :]
        out = super(MaskedBatchNorm1d, self).forward(inputs, mask)
        if self.channels_last:
            out = out.permute(0, 2, 1)
        return out


class MaskedBatchNorm1dV2(nn.BatchNorm1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, mask=None):
        """
        Args:
            x: NCL
            mask: NL where 1 is non-mask, 0 is mask
        """
        if mask is None:
            return super().forward(x)

        # NCL -> NLC -> (NL)C
        N, C, L = x.shape
        x = x.permute(0, 2, 1).reshape(-1, C)
        mask = mask.view(-1)
        x[mask] = super().forward(x[mask])
        x = x.view((N, L, C)).permute(0, 2, 1)  # NLC -> NCL
        return x


class LayerScale(nn.Module):
    """
    Computes an affine transformation y = x * scale + bias, either learned via adaptive weights, or fixed.
    Efficient alternative to LayerNorm where we can avoid computing the mean and variance of the input, and
    just rescale the output of the previous layer.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([1.0] * dim)[None, None, :])
        self.bias = torch.nn.Parameter(torch.tensor([0.0] * dim)[None, None, :])

    def forward(self, x):
        return x * self.scale + self.bias


class IBNorm3d(nn.Module):
    """
    3D version of Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Ref: https://github.com/XingangPan/IBN-Net/blob/master/ibnnet/modules.py

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio=0.5):
        super().__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm3d(self.half, affine=True)
        self.BN = nn.BatchNorm3d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class IBNorm2d(nn.Module):
    """
    Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Ref: https://github.com/XingangPan/IBN-Net/blob/master/ibnnet/modules.py

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio=0.5):
        super().__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class IBNorm1d(nn.Module):

    def __init__(self, planes, ratio=0.5):
        super().__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm1d(self.half, affine=True)
        self.BN = nn.BatchNorm1d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class GhostBatchNorm1d(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(
        self,
        input_dim,
        vbs=None,
        splits=None,
        eps=1e-05,
        momentum=0.01,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        """
        Args:
            input_dim: input channel dim size
            vbs: virtual batch size
        """
        super().__init__()
        self.input_dim = input_dim
        if vbs is not None:
            assert splits is None
        elif splits is not None:
            assert vbs is None
        else:
            raise ValueError
        self.vbs = vbs
        self.bn = nn.BatchNorm1d(
            self.input_dim,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        chunks = x.chunk(int(math.ceil(x.size(0) / self.vbs)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)


class ChannelLastBatchNorm1d(nn.BatchNorm1d):
    """
    Input: NLC
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class ChannelLastBatchNorm2d(nn.BatchNorm2d):
    """
    Input: NHWC
    """

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = super().forward(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        return x


class ChannelLastBatchNorm3d(nn.BatchNorm1d):
    """
    Input: NDHWC or NTHWC
    """

    def forward(self, x):
        # NTHWC -> NCTHW
        x = x.permute(0, 4, 1, 2, 3)
        x = super().forward(x)
        # NCTHW -> NTHWC
        x = x.permute(0, 2, 3, 4, 1)
        return x


class ChannelLastMaskedBatchNorm1d(MaskedBatchNorm1d):
    """
    Quick implementation and lack of optimization.
    Input: NLC
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class GeneralizedLayerNorm(nn.Module):
    """
    LayerNorm that supports 'channels_last' (NLC / NHWC / NDHWC) or 'channels_first' (NCL / NCHW / NCDHW).
    Normalizes across channels only (C), not across spatial dims.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
        bias=True,
        device=None,
        dtype=None,
        data_format="channel_last",
        fp32=True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if data_format not in ("channel_last", "channel_first"):
            raise ValueError("data_format must be 'channel_last' or 'channel_first'")
        self.data_format = data_format
        # allow int or tuple input; store as a tuple as PyTorch expects
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        elif isinstance(normalized_shape, (tuple, list)):
            self.normalized_shape = tuple(normalized_shape)
        else:
            raise TypeError("normalized_shape must be int or tuple/list")

        C = self.normalized_shape[-1]  # we normalize over channels only
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.fp32 = fp32
        if elementwise_affine:
            # gamma (weight) and beta (bias), shape (C,)
            self.weight = nn.Parameter(torch.ones(C, **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros(C, **factory_kwargs), requires_grad=bias
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            # expect x[..., C]; normalized_shape should be (C,)
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )

        # channels_first: x is (N, C, *spatial)
        # normalize across channel dim only, per spatial location
        assert x.dim() >= 3, f"Expected N,C,*spatial; got {tuple(x.shape)}"
        C = x.size(1)
        assert C == (
            self.weight.numel() if self.weight is not None else self.normalized_shape[0]
        ), "Channel count mismatch between x and LayerNorm"

        if self.fp32:
            # upcasts the activations to FP32 just for the mean/var math -> Stability in mixed precision
            xf = x.float()
        else:
            xf = x
        mean = xf.mean(dim=1, keepdim=True).type_as(x)  # (N,1,*)
        var = xf.var(dim=1, keepdim=True, unbiased=False).type_as(x)  # (N,1,*)
        x = (x - mean) * torch.rsqrt(var + self.eps)

        if self.elementwise_affine:
            shape = (1, C) + (1,) * (x.dim() - 2)
            x = x * self.weight.view(*shape).type_as(x) + self.bias.view(
                *shape
            ).type_as(x)
        return x

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}, "
            "data_format={data_format}, "
            "fp32={fp32}".format(**self.__dict__)
        )


class ChannelFirstLayerNorm1d(nn.LayerNorm):
    """
    Input: NCL
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class SlowChannelFirstLayerNorm2d(nn.LayerNorm):
    """DEPRECATED, use `timm.layers.norm.LayerNorm2d` instead
    Input: NCHW
    """

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x


ChannelFirstLayerNorm2d = timm_norm.LayerNorm2d


class ChannelFirstLayerNorm3d(nn.LayerNorm):
    """
    Input: NCDHW or NCTHW
    """

    def forward(self, x):
        # x: (B, C, D, H, W) → (B, D, H, W, C)
        x = x.permute(0, 2, 3, 4, 1)  # NCHW -> NHWC
        x = super().forward(x)
        # Back to (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)  # NHWC -> NCHW
        return x


class ChannelLastInstanceNorm1d(nn.InstanceNorm1d):
    """
    Input: NCHW
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)  # NCHW -> NHWC
        x = super().forward(x)
        x = x.permute(0, 2, 1)  # NHWC -> NCHW
        return x


# @TODO: support GroupNorm
# @TODO: support SplitBatchNorm
# @TODO: support GhostBatchNorm
# @TODO: flexible args & kwargs
def get_norm(name, *args, channel_first=True, **kwargs):
    if name == "identity":
        return nn.Identity
    elif name == 'batchnorm':
        return nn.BatchNorm1d
    elif name == 'layernorm':
        return nn.LayerNorm
    elif name == "batchnorm_1d":
        if channel_first:
            # Channel-first BatchNorm1d, NCL
            return partial(nn.BatchNorm1d, *args, **kwargs)
        else:
            # Channel-last BatchNorm1d, NLC
            return partial(ChannelLastBatchNorm1d, *args, **kwargs)
    elif name == "batchnorm_2d":
        if channel_first:
            # Channel-first BatchNorm2d, NCL
            return partial(nn.BatchNorm2d, *args, **kwargs)
        else:
            # Channel-last BatchNorm1d, NLC
            return partial(ChannelLastBatchNorm2d, *args, **kwargs)

    elif name == "batchnorm_3d":
        if channel_first:
            # Channel-first BatchNorm3d, NCDHW or NCTHW
            return partial(nn.BatchNorm3d, *args, **kwargs)
        else:
            # Channel-last BatchNorm3d, NDHWC or NTHWC
            return partial(ChannelLastBatchNorm3d, *args, **kwargs)
    elif name == "masked_batchnorm_1d":
        if channel_first:
            # Channel-first Masked BatchNorm1d, NCL
            # mask=0 is pad, mask=1 is non-pad
            return partial(MaskedBatchNorm1d, *args, **kwargs)
        else:
            # Channel-last Masked BatchNorm1d, NLC
            # mask=0 is pad, mask=1 is non-pad
            return partial(ChannelLastMaskedBatchNorm1d, *args, **kwargs)
    elif name == "masked_batchnorm_1d_v2":
        if channel_first:
            # Channel-first Masked BatchNorm1d, NCL
            # mask=0 is pad, mask=1 is non-pad
            return partial(MaskedBatchNorm1dV2, *args, **kwargs)
        raise NotImplementedError
    elif name == "layernorm_1d":
        if channel_first:
            # Channel-first LayerNorm1d, NCL
            return partial(ChannelFirstLayerNorm1d, *args, **kwargs)
        else:
            # Channel-last LayerNorm1d, NLC
            return partial(nn.LayerNorm, *args, **kwargs)
    elif name == "layernorm_2d":
        if channel_first:
            # Channel-first LayerNorm2d, NCHW
            return partial(ChannelFirstLayerNorm2d, *args, **kwargs)
        else:
            # Channel-last Layernorm2d, NHWC
            return partial(nn.LayerNorm, *args, **kwargs)
    elif name == "layernorm_3d":
        if channel_first:
            # Channel-first LayerNorm3d, NCDHW or NCTHW
            return partial(ChannelFirstLayerNorm3d, *args, **kwargs)
        else:
            # Channel-last Layernorm3d, NDHWC or NTHWC
            return partial(nn.LayerNorm, *args, **kwargs)
    elif name == "instancenorm_1d":
        if channel_first:
            # Channel-first InstanceNorm1d, NCL
            return partial(nn.InstanceNorm1d, *args, **kwargs)
        else:
            # Channel-last InstanceNorm1d, NLC
            return partial(ChannelLastInstanceNorm1d, *args, **kwargs)
    elif name == "ibnorm_1d":
        if channel_first:
            # Channel-first IBN, NCL
            return partial(IBNorm1d, *args, **kwargs)
        # Channel-last IBN, NLC
        raise NotImplementedError
    elif name == "ibnorm_2d":
        if channel_first:
            # Channel-first IBN, NCL
            return partial(IBNorm2d, *args, **kwargs)
        # Channel-last IBN, NLC
        raise NotImplementedError
    elif name == "ibnorm_3d":
        if channel_first:
            # Channel-first IBN, NCL
            return partial(IBNorm3d, *args, **kwargs)
        # Channel-last IBN, NLC
        raise NotImplementedError
    elif name == "ghostbatchnorm_1d":
        if channel_first:
            return partial(GhostBatchNorm1d, *args, **kwargs)
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid norm name {name}')


def _module_device_dtype(m: nn.Module):
    for t in m.state_dict().values():
        if isinstance(t, torch.Tensor):
            return t.device, t.dtype
    return torch.device("cpu"), torch.float32


def replace_norm_inplace(module, src_class, dst_class, verbose=True, **dst_norm_kwargs):
    """
    Stupid and not well-refactored function.
    In-place: recursively replace all BatchNorm{1,2,3}d with:
      - InstanceNorm{1,2,3}d (keeps track_running_stats if given),
      - GroupNorm (num_groups auto-chosen if not provided),
      - ChannelFirstLayerNorm (channel-wise LN suitable for NCHW/NCDHW).
    """
    assert src_class in [
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
    ], "Currently support convert BatchNorm only"

    dim = {
        nn.BatchNorm1d: 1,
        nn.BatchNorm2d: 2,
        nn.BatchNorm3d: 3,
    }[src_class]
    for name, m in list(module.named_children()):
        # Recurse first
        replace_norm_inplace(m, src_class, dst_class, verbose=False, **dst_norm_kwargs)

        # Then possibly replace this child
        if not isinstance(m, src_class):
            continue

        C = m.num_features
        device, dtype = _module_device_dtype(m)

        if dst_class == "batchnorm":
            cls = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[dim]
            new = cls(
                C,
                eps=dst_norm_kwargs.get("eps", 1e-5),
                momentum=dst_norm_kwargs.get("momentum", 0.1),
                affine=dst_norm_kwargs.get("affine", True),
                track_running_stats=dst_norm_kwargs.get("track_running_stats", True),
                device=device,
                dtype=dtype,
            )
        elif dst_class == "instancenorm":
            cls = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[
                dim
            ]
            new = cls(
                C,
                eps=dst_norm_kwargs.get("eps", 1e-5),
                momentum=dst_norm_kwargs.get("momentum", 0.1),
                affine=dst_norm_kwargs.get("affine", False),
                track_running_stats=dst_norm_kwargs.get("track_running_stats", False),
                device=device,
                dtype=dtype,
            )
        elif dst_class == "groupnorm":
            new = nn.GroupNorm(
                num_groups=dst_norm_kwargs["num_groups"],
                num_channels=C,
                eps=dst_norm_kwargs.get("eps", 1e-5),
                affine=dst_norm_kwargs.get("affine", True),
                device=device,
                dtype=dtype,
            )
        elif dst_class == "layernorm":
            # cls = {
            #     1: ChannelFirstLayerNorm1d,
            #     2: ChannelFirstLayerNorm2d,
            #     3: ChannelFirstLayerNorm3d,
            # }[dim]
            cls = partial(GeneralizedLayerNorm, data_format="channel_first")
            new = cls(
                C,
                eps=dst_norm_kwargs.get("eps", 1e-5),
                elementwise_affine=dst_norm_kwargs.get("elementwise_affine", True),
                bias=dst_norm_kwargs.get("bias", True),
                device=device,
                dtype=dtype,
                fp32=dst_norm_kwargs.get("fp32", True),
            )
        elif dst_class == "ibnorm":
            cls = {1: IBNorm1d, 2: IBNorm2d, 3: IBNorm3d}[dim]
            new = cls(C, ratio=0.5)
        else:
            raise ValueError(
                "norm must be one of: 'instancenorm' | 'groupnorm' | 'layernorm'"
            )

        # Move to same device/dtype
        # new = new.to(device)

        # Try copying gamma/beta if present and shapes match
        with torch.no_grad():
            if (
                getattr(m, "weight", None) is not None
                and getattr(new, "weight", None) is not None
            ):
                if new.weight.shape == m.weight.shape:
                    logger.info(
                        "replace_norm(): Transfering weight of shape %s, dtype=%s",
                        m.weight.shape,
                        m.weight.dtype,
                    )
                    new.weight.copy_(m.weight)
            if (
                getattr(m, "bias", None) is not None
                and getattr(new, "bias", None) is not None
            ):
                if new.bias.shape == m.bias.shape:
                    logger.info(
                        "replace_norm(): Transfering bias of shape %s, dtype=%s",
                        m.bias.shape,
                        m.bias.dtype,
                    )
                    new.bias.copy_(m.bias)
        # modify inplace
        module._modules[name] = new
        logger.info(
            "replace_norm(): %s (%s --> %s)",
            name,
            m.__class__.__name__,
            new.__class__.__name__,
        )
    return module
