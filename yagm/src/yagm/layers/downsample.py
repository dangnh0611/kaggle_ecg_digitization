"""These modules are borrowed/modified from timm."""

from typing import Tuple

from torch import nn

__all__ = ["Downsample1d"]


# @TODO - Can SAME padding for given args be done statically?
def _is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    """Ref: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/padding.py"""
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def _get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    """Calculate symmetric padding for a convolution."""
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def _get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == "same":
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if _is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = _get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == "valid":
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = _get_padding(kernel_size, **kwargs)
    return padding, dynamic


def _create_pool1d(pool_type, kernel_size, stride=None, **kwargs):
    """Tweak from timm.
    Ref: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pool2d_same.py#L56
    """
    stride = stride or kernel_size
    padding = kwargs.pop("padding", "")
    padding, is_dynamic = _get_padding_value(
        padding, kernel_size, stride=stride, **kwargs
    )
    if is_dynamic:
        # never goes here
        raise NotImplementedError
        if pool_type == "avg":
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == "max":
            return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
    else:
        if pool_type == "avg":
            return nn.AvgPool1d(kernel_size, stride=stride, padding=padding, **kwargs)
        elif pool_type == "max":
            return nn.MaxPool1d(kernel_size, stride=stride, padding=padding, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"


# tweak from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/maxxvit.py#L303
class Downsample1d(nn.Module):
    """Tweak from timm.
    A downsample pooling module supporting several maxpool and avgpool modes
    * 'max' - MaxPool2d w/ kernel_size 3, stride 2, padding 1
    * 'max2' - MaxPool2d w/ kernel_size = stride = 2
    * 'avg' - AvgPool2d w/ kernel_size 3, stride 2, padding 1
    * 'avg2' - AvgPool2d w/ kernel_size = stride = 2
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        pool_type: str = "avg2",
        padding: str = "",
        bias: bool = True,
    ):
        super().__init__()
        assert pool_type in ("max", "max2", "avg", "avg2", "conv", "conv2")
        if pool_type == "max":
            self.pool = _create_pool1d(
                "max", kernel_size=3, stride=2, padding=padding or 1
            )
        elif pool_type == "max2":
            self.pool = _create_pool1d(
                "max", 2, padding=padding or 0
            )  # kernel_size == stride == 2
        elif pool_type == "avg":
            self.pool = _create_pool1d(
                "avg",
                kernel_size=3,
                stride=2,
                count_include_pad=False,
                padding=padding or 1,
            )
        elif pool_type == "avg2":
            self.pool = _create_pool1d("avg", 2, padding=padding or 0)
        elif pool_type == "conv":
            self.pool = nn.Conv1d(dim, dim_out, kernel_size=3, stride=2, padding=0)
        elif pool_type == "conv2":
            self.pool = nn.Conv1d(dim, dim_out, kernel_size=2, stride=2, padding=1)
        else:
            raise NotImplementedError

        if dim != dim_out and "conv" not in pool_type:
            self.expand = nn.Conv1d(dim, dim_out, 1, bias=bias)
        else:
            self.expand = nn.Identity()

    def forward(self, x):
        x = self.pool(x)  # spatial downsample
        x = self.expand(x)  # expand chs
        return x
