from torch import nn

__all__ = ["DepthwiseConv1d", "PointwiseConv1dNormAct", "DepthwiseConv1dNormAct"]


class DepthwiseConv1d(nn.Conv1d):
    """Depthwise Convolution 1D, which keep sequence length un-changed (`same` padding)"""

    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=True,
    ):
        assert stride <= 2
        if isinstance(kernel_size, int) and isinstance(dilation, int):
            padding = (kernel_size - 1) // 2 * dilation
        else:
            raise ValueError
        super().__init__(
            in_channels,
            depth_multiplier * in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )


class PointwiseConv1dNormAct(nn.Module):

    def __init__(self, in_dim, out_dim, norm_layer=None, act_layer=None, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1, bias=bias)
        self.norm = norm_layer(out_dim) if norm_layer is not None else nn.Identity()
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DepthwiseConv1dNormAct(nn.Module):

    def __init__(
        self, in_dim, kernel_size, stride, norm_layer, act_layer=None, bias=True
    ):
        super().__init__()
        self.conv = DepthwiseConv1d(
            in_dim,
            depth_multiplier=1,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )
        self.norm = norm_layer(in_dim)
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
