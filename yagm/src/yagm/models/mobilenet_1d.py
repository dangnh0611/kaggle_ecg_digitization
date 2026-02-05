from torch import nn
from torchvision.ops import StochasticDepth

from yagm.layers.common import make_divisible
from yagm.layers.depthwise_conv import DepthwiseConv1dNormAct, PointwiseConv1dNormAct
from yagm.layers.downsample import Downsample1d
from yagm.layers.external_attention import ECAAttention1d, SEAttention1d


class MBBlock1d(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride,
        expand_ratio,
        downsample="avg",
        dropout=0.0,
        droppath=0.0,
        norm_layer=None,
        act_layer=None,
        attn_name=None,
    ):
        super(MBBlock1d, self).__init__()
        self.stride = stride

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if act_layer is None:
            act_layer = nn.ReLU

        hidden_dim = int(round(in_dim * expand_ratio))

        self.use_shortcut = True
        if self.stride == 1:
            assert in_dim == out_dim
            self.shortcut = nn.Identity()
        elif self.stride == 2 and downsample != "none":
            self.shortcut = Downsample1d(in_dim, out_dim, downsample, bias=True)
        elif self.stride == 2 and downsample == "none":
            self.use_shortcut = False
        else:
            raise ValueError

        if expand_ratio != 1:
            # expand pw
            self.expand_pw = PointwiseConv1dNormAct(
                in_dim, hidden_dim, norm_layer=norm_layer, act_layer=act_layer
            )
        else:
            raise NotImplementedError
        # depth-wise
        self.dw = DepthwiseConv1dNormAct(
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        if attn_name == "SE":
            se_hidden_dim = make_divisible(hidden_dim // 4, 8)
            # mobilenetv3 use HardSigmoid instead of Sigmoid
            # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py#L52C35-L52C35
            self.attn = SEAttention1d(
                hidden_dim, se_hidden_dim, hidden_act=nn.ReLU, scale_act=nn.Hardsigmoid
            )
        elif attn_name == "ECA":
            self.attn = ECAAttention1d(kernel_size=5)
        elif attn_name is None:
            self.attn = None
        else:
            raise ValueError(f"Invalid channel attention {attn_name}")
        # point-wise + norm, no activation (linear)
        self.pw = PointwiseConv1dNormAct(
            hidden_dim, out_dim, norm_layer=norm_layer, act_layer=None
        )
        self.droppath = StochasticDepth(p=droppath, mode="row")
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.use_shortcut:
            skip = self.shortcut(x)
        x = self.expand_pw(x)
        x = self.dw(x)
        if self.attn is not None:
            x = self.attn(x)
        x = self.pw(x)
        x = self.droppath(self.dropout(x))
        if self.use_shortcut:
            x = x + skip
        return x

    def init_weights(self, m):
        # weight initialization
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# less activations and norms
# https://www.kaggle.com/code/hoyso48/1st-place-solution-inference
class MBBlock1dV2(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride,
        expand_ratio,
        downsample="avg",
        dropout=0.0,
        droppath=0.0,
        norm_layer=None,
        act_layer=None,
        attn_name=None,
    ):
        super(MBBlock1dV2, self).__init__()
        self.stride = stride

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if act_layer is None:
            act_layer = nn.ReLU

        hidden_dim = int(round(in_dim * expand_ratio))

        if self.stride == 1:
            assert in_dim == out_dim
            self.shortcut = nn.Identity()
        elif self.stride == 2:
            self.shortcut = Downsample1d(in_dim, out_dim, downsample, bias=True)
        else:
            raise ValueError

        if expand_ratio != 1:
            # expand pw
            # with bias + activation, no norm
            self.expand_pw = PointwiseConv1dNormAct(
                in_dim, hidden_dim, norm_layer=None, act_layer=act_layer, bias=True
            )
        else:
            raise NotImplementedError
        # depth-wise
        # ori: no bias + batchnorm, no activation
        # this: bias + BN/LN/GN, no activation
        # bias could be rebundant but no impact on performance -> keep bias=True
        self.dw = DepthwiseConv1dNormAct(
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            norm_layer=norm_layer,
            act_layer=None,
            bias=True,
        )
        if attn_name == "SE":
            se_hidden_dim = make_divisible(hidden_dim // 4, 8)
            # mobilenetv3 use HardSigmoid instead of Sigmoid
            # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py#L52C35-L52C35
            self.attn = SEAttention1d(
                hidden_dim, se_hidden_dim, hidden_act=nn.ReLU, scale_act=nn.Hardsigmoid
            )
        elif attn_name == "ECA":
            self.attn = ECAAttention1d(kernel_size=5)
        elif attn_name is None:
            self.attn = None
        else:
            raise ValueError(f"Invalid channel attention {attn_name}")
        # point-wise, no norm, no activation (linear)
        self.pw = PointwiseConv1dNormAct(
            hidden_dim, out_dim, norm_layer=None, act_layer=None, bias=True
        )
        self.droppath = StochasticDepth(p=droppath, mode="row")
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        skip = self.shortcut(x)
        x = self.expand_pw(x)
        x = self.dw(x)
        if self.attn is not None:
            x = self.attn(x)
        x = self.pw(x)
        x = self.droppath(self.dropout(x))
        x = x + skip
        return x

    def init_weights(self, m):
        # weight initialization
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
