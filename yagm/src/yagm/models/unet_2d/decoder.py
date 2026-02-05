import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d
from yagm.layers.activation import get_act
from yagm.layers.common import ConvNormAct2d, Interpolate
from yagm.layers.conv_factory import CONV2D_TYPES, DCNv4Wrapper, get_conv_layer
from yagm.layers.external_attention import ExternalAttention
from yagm.layers.norms import get_norm

logger = logging.getLogger(__name__)


def compute_same_pad(k: int) -> Tuple[int, int, int, int]:
    """
    Return (pad_left, pad_right, pad_top, pad_bottom).
    Uses the common convention: bias extra padding to the right/bottom.
    """
    if k % 2 == 1:
        p = k // 2
        return (p, p, p, p)
    else:
        # Total pad = k - 1
        # Pad before = (k - 1) // 2
        # Pad after = (k - 1) - Pad before
        pad_before = (k - 1) // 2
        pad_after = (k - 1) - pad_before

        # For K=2: pad_before=0, pad_after=1. Returns (0, 1, 0, 1)
        return (pad_before, pad_after, pad_before, pad_after)


class DepthwiseGaussianBlur(nn.Module):
    """
    Depthwise separable fixed Gaussian blur with precomputed depthwise kernel.

    - channels: number of channels (depthwise groups)
    - ksize: int (odd or even)
    - sigma: None -> auto gaussian sigma; numeric -> gaussian sigma
             (passing 0 or "box" is not supported and raises ValueError)
    - pad_mode: 'replicate' by default
    """

    def __init__(
        self,
        channels: int,
        ksize: int = 3,
        pad_mode: str = "replicate",
        sigma: Optional[float] = None,
    ):
        super().__init__()
        assert pad_mode in ["reflect", "replicate"]

        if ksize < 1 or not isinstance(ksize, int):
            raise ValueError("kernel_size must be a positive integer")
        if channels < 1 or not isinstance(channels, int):
            raise ValueError("channels must be a positive integer")

        self.channels = channels
        self.pad_mode = pad_mode
        self._pad = compute_same_pad(ksize)  # (left, right, top, bottom)

        # Build 1D gaussian kernel
        k1d = self._make_gaussian_kernel(ksize, sigma)
        # Build 2D kernel (outer product) and expand to depthwise shape (channels,1,k,k)
        k2d_base = torch.outer(k1d, k1d).view(1, 1, ksize, ksize)  # (1,1,k,k)
        k2d = k2d_base.repeat(self.channels, 1, 1, 1)  # (channels,1,k,k)
        self.register_buffer("_k2d", k2d)  # precomputed full depthwise 2D kernel

    def _make_gaussian_kernel(
        self, kernel_size: int, sigma: Optional[float] = None
    ) -> torch.Tensor:
        if sigma is None:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        coords = (
            torch.arange(kernel_size, dtype=dtype, device=device)
            - (kernel_size - 1) / 2.0
        )
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: blurred x, same H,W
        """
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Input has {x.shape[1]} channels but blur was created with channels={self.channels}"
            )
        x = F.pad(x, self._pad, mode=self.pad_mode)
        # depth-wise conv
        return F.conv2d(x, self._k2d.to(dtype=x.dtype), groups=self.channels)


class MeanBlur(nn.Module):
    """Box/Mean Filter
    Applies padding then a sliding mean with kernel_size (stride=1), preserving input spatial size.

    Args:
        ksize: size of the square mean filter (int >= 1). Can be odd or even.
        padding_mode: one of padding modes accepted by torch.nn.functional.pad:
                      'reflect', 'replicate', 'constant', 'circular'. Default 'replicate'.
        value: fill value used when padding_mode == 'constant' (ignored otherwise).
    """

    def __init__(self, ksize: int = 2, pad_mode: str = "replicate", value: float = 0.0):
        super().__init__()
        if not isinstance(ksize, int) or ksize < 1:
            raise ValueError("kernel_size must be an integer >= 1")
        if pad_mode not in ("reflect", "replicate", "constant", "circular"):
            raise ValueError(
                "padding_mode must be one of 'reflect','replicate','constant','circular'"
            )

        self.kernel_size = ksize
        self.pad_mode = pad_mode
        self.value = float(value)

        # Precompute padding tuple for F.pad: (pad_left, pad_right, pad_top, pad_bottom)
        self._pad = compute_same_pad(ksize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: blurred tensor, same shape (B, C, H, W)
        """
        padded = F.pad(x, self._pad, mode=self.pad_mode, value=self.value)
        # sliding mean using avg_pool2d with stride=1
        out = F.avg_pool2d(padded, kernel_size=self.kernel_size, stride=1, padding=0)
        return out


# -------------------------
# PixelShuffleICNR block
# -------------------------
class PixelShuffleICNR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        upscale_factor: int = 2,
        norm_layer: Union[str, Callable[[int], nn.Module]] = LayerNorm2d,
        act_layer: Union[str, Callable[[], nn.Module]] = nn.GELU,
        conv_layer=nn.Conv2d,
        post_conv: bool = False,
        blur: str = "mean",
        blur_ksize: int = 2,
        blur_pad_mode="replicate",
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.upsample_factor = upscale_factor
        self.out_channels = out_channels

        self.conv = ConvNormAct2d(
            in_channels,
            out_channels * (upscale_factor * upscale_factor),
            kernel_size=1,
            bias=True,
            norm_layer=norm_layer,
            act_layer=act_layer,
            conv_layer=conv_layer,
        )

        # ICNR conv weight initialization
        self._icnr_init_conv(
            self.conv[0], scale=upscale_factor, init=nn.init.kaiming_normal_
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.post_conv = (
            ConvNormAct2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                norm_layer=norm_layer,
                act_layer=act_layer or nn.GELU,
                conv_layer=conv_layer,
            )
            if post_conv
            else nn.Identity()
        )

        if blur is None or blur == "identity":
            self.blur = nn.Identity()
        elif blur == "box" or blur == "mean":
            self.blur = MeanBlur(blur_ksize, blur_pad_mode)
        elif blur == "gaussian":
            self.blur = DepthwiseGaussianBlur(
                out_channels, ksize=blur_ksize, pad_mode=blur_pad_mode
            )
        else:
            raise ValueError

    def _icnr_init_conv(
        self, conv: nn.Conv2d, scale: int = 2, init=nn.init.kaiming_normal_
    ):
        """
        Initialize conv.weight (shape [out_ch, in_ch, kh, kw]) using ICNR.
        conv must be an nn.Conv2d with out_channels divisible by scale^2.
        """
        if not isinstance(conv, CONV2D_TYPES):
            raise TypeError(f"conv must be a 2D convolution layer: {CONV2D_TYPES}")

        w = conv.weight  # Parameter or tensor
        out_ch, in_ch, kh, kw = w.shape
        if out_ch % (scale * scale) != 0:
            raise ValueError("out_ch must be divisible by scale^2")

        new_out = out_ch // (scale * scale)
        # create subkernel and initialize it
        subkernel = torch.zeros(
            (new_out, in_ch, kh, kw), device=w.device, dtype=w.dtype
        )
        init(subkernel)
        # repeat subkernel s^2 times along output dim
        repeated = subkernel.repeat_interleave(scale * scale, dim=0)
        with torch.no_grad():
            w.copy_(repeated)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.post_conv(x)
        x = self.blur(x)
        return x


class UnetDecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        attn_type: Optional[str] = None,
        upscale_type: str = "interpolate",  #  interpolate | pixelshuffle
        upscale_factor: Tuple[int, int] = (2, 2),
        norm_layer: Union[str, Callable[[], nn.Module]] = "layernorm",
        act_layer: Union[type, Callable[[], nn.Module]] = "gelu",
        conv_layer: Union[type, Callable[[], nn.Module]] = nn.Conv2d,
        interpolation_mode: str = "nearest",
        interpolation_antialias: bool = False,
        shuffle_blur="mean",  # pixelshuffle specific
        shuffle_blur_ksize=2,  # pixelshuffle specific
        shuffle_post_conv=False,  # pixelshuffle specific
        use_residual: bool = False,
        norm_after_conv=True,
        norm_on_skip=True,
        norm_on_cat=False,
        act_on_cat=False,
    ):
        """
        Args:
            norm_after_conv: conv -> norm -> act if True, else conv -> act only.
            norm_on_skip: whether to apply normalization on skip features.
                Should be disable if skip features from encoder already normalized
            norm_on_cat: experimental, used nowhere :D
            act_on_cat: some strange implementation apply activation right after concatenated feature up+skip
                ref: https://github.com/DrHB/2nd-place-contrails/blob/master/src_inference1/layers.py#L70
        """
        super().__init__()
        assert upscale_type in ["interpolate", "pixelshuffle"]
        self.upscale_type = upscale_type
        self.upscale_factor = upscale_factor
        self.interpolation_mode = interpolation_mode
        self.interpolation_antialias = interpolation_antialias
        self.norm_on_skip = norm_on_skip
        self.norm_on_cat = norm_on_cat
        self.act_on_cat = act_on_cat

        # Upsample type
        if self.upscale_type == "pixelshuffle":
            if upscale_factor[0] != upscale_factor[1]:
                logger.warning(
                    "Using PixelShuffle with non-uniform upscale_factor of %s",
                    upscale_factor,
                )
            _scale = max(upscale_factor)
            # spatial resolution is increase by _scale**2
            # it's safe to reduce the output channels by _scale
            up_out_channels = max(in_channels // _scale, 1)
            self.upsample = PixelShuffleICNR(
                in_channels,
                out_channels=up_out_channels,
                upscale_factor=_scale,
                norm_layer=norm_layer,
                act_layer=act_layer,
                conv_layer=conv_layer,
                post_conv=shuffle_post_conv,
                blur=shuffle_blur,
                blur_ksize=shuffle_blur_ksize,
            )
        elif self.upscale_type == "interpolate":
            up_out_channels = in_channels
            self.upsample = Interpolate(
                scale_factor=None,  # passed shape on forward() call
                mode=interpolation_mode,
                antialias=interpolation_antialias,
            )
        else:
            raise ValueError("upsample_type must be 'interpolate' or 'pixelshuffle'")

        # attention on concatenated up+skip features
        self.attention1 = ExternalAttention(
            attn_type, in_channels=up_out_channels + skip_channels
        )

        # skip normalization
        if skip_channels > 0 and norm_on_skip:
            self.skip_norm = norm_layer(skip_channels)
        else:
            self.skip_norm = nn.Identity()

        # normalization right after feature concat up+skip
        if norm_on_cat:
            self.cat_norm = norm_layer(up_out_channels + skip_channels)
        else:
            self.cat_norm = nn.Identity()

        # activation right after feature concat up+skip (posible after an additional norm)
        # cat -> norm -> act
        if act_on_cat:
            self.cat_act = act_layer()
        else:
            self.cat_act = nn.Identity()

        # conv1 and conv2 using ConvNormAct2d
        # SMP use conv2d -> norm -> act by default, where norm is default to BatchNorm2d
        # but some implementation prevent using norm in DecoderBlock, just conv2d -> act
        # https://github.com/DrHB/2nd-place-contrails/blob/master/src_inference1/layers.py#L70
        self.conv1 = ConvNormAct2d(
            up_out_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_layer=norm_layer if norm_after_conv else None,
            act_layer=act_layer,
            conv_layer=conv_layer,
        )
        self.conv2 = ConvNormAct2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_layer=norm_layer if norm_after_conv else None,
            act_layer=act_layer,
            conv_layer=conv_layer,
        )

        # attention after fused features after convolutions
        self.attention2 = ExternalAttention(attn_type, in_channels=out_channels)

        # residual branch
        self.use_residual = use_residual
        if use_residual:
            if (up_out_channels + skip_channels) != out_channels:
                self.res_conv = ConvNormAct2d(
                    up_out_channels + skip_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                    norm_layer=norm_layer,
                    act_layer=None,
                    conv_layer=conv_layer,
                )
            else:
                self.res_conv = nn.Identity()

        # initialize conv weights for ConvNormAct2d first convs
        _conv_layers = [self.conv1[0], self.conv2[0]]
        try:
            _conv_layers.append(self.res_conv[0])
        except:
            pass
        for m in _conv_layers:
            assert isinstance(m, CONV2D_TYPES)
            if not isinstance(m, DCNv4Wrapper):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                logger.info(
                    "Skip init weight of UnetDecoderBlock's 2 convolution layer"
                )

    def forward(
        self,
        feature_map: torch.Tensor,
        target_height: int,
        target_width: int,
        skip_connection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Upsample feature_map to target
        if self.upscale_type == "pixelshuffle":
            # if current resolution not compatible with pixelshuffle
            # just interpolate to a suitable resolution
            hs, ws = self.upscale_factor
            pre_h = target_height // hs
            pre_w = target_width // ws
            if feature_map.shape[2] != pre_h or feature_map.shape[3] != pre_w:
                logger.warning(
                    "Using PixelShuffle but resolution is imcompatible, using F.interpolate first."
                )
                feature_map = F.interpolate(
                    feature_map,
                    size=(pre_h, pre_w),
                    mode=self.interpolation_mode,
                    antialias=self.interpolation_antialias,
                )
            # apply pixelshuffle
            up_out = self.upsample(feature_map)
        else:
            up_out = self.upsample(feature_map, size=(target_height, target_width))

        # skip processing + attention
        s = skip_connection
        if s is not None:
            # if s.shape[2] != target_height or s.shape[3] != target_width:
            #     s = F.interpolate(
            #         s, size=(target_height, target_width), mode=self.interpolation_mode
            #     )
            s = self.skip_norm(s)
            # @TODO - another implementation which apply an activation right after torch.cat
            # https://github.com/DrHB/2nd-place-contrails/blob/master/src_inference1/layers.py#L70
            cat = torch.cat([up_out, s], dim=1)
            cat = self.cat_act(self.cat_norm(cat))
            cat = self.attention1(cat)
        else:
            cat = up_out

        x = self.conv2(self.conv1(cat))
        x = self.attention2(x)

        if self.use_residual:
            # residual connection from concatenated feature to current feature (after 2 conv blocks)
            res = self.res_conv(cat)
            x = x + res
        return x


# -------------------------
# Center block (unchanged SMP style)
# -------------------------
class UnetCenterBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Optional[Union[str, Callable[[], nn.Module]]] = "layernorm",
        act_layer: Optional[Union[str, Callable[[], nn.Module]]] = "gelu",
        conv_layer: Optional[Union[str, Callable[[], nn.Module]]] = nn.Conv2d,
        norm_after_conv: bool = True,
    ):
        # fallback simple implementation
        super().__init__(
            ConvNormAct2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_layer=norm_layer if norm_after_conv else None,
                act_layer=act_layer,
                conv_layer=conv_layer,
            ),
            ConvNormAct2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_layer=norm_layer if norm_after_conv else None,
                act_layer=act_layer,
                conv_layer=conv_layer,
            ),
        )


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],  # fine to coarse
        encoder_strides: Sequence[Union[int, tuple[int, int]]],  # fine to coarse
        decoder_channels: Sequence[int],  # coarse to fine
        n_blocks: int = 5,
        upscale_type: str = "interpolate",  # interpolate | pixelshuffle
        norm_layer: Optional[Union[str, Callable[[int], nn.Module]]] = "layernorm",
        act_layer: Optional[Union[str, Callable[[], nn.Module]]] = "gelu",
        conv_layer: Optional[Union[str, Callable[[], nn.Module]]] = "conv2d",
        conv_kwargs: Optional[Dict[str, Any]] = None,
        attn_type: Optional[str] = None,
        # interpolate specific kwargs
        interpolation_mode: str = "nearest",
        interpolation_antialias: bool = False,
        # pixelshuffle specific kwargs
        shuffle_blur="mean",  # pixelshuffle specific
        shuffle_blur_ksize=2,  # pixelshuffle specific
        shuffle_post_conv=False,  # pixelshuffle specific
        # fine-grained customizations
        center_block: bool = False,
        skip_on_finest=False,  # whether to add skip connection from original resolution
        use_residual: bool = False,
        norm_after_conv: bool = True,
        norm_on_skip: bool = True,
        norm_on_cat: bool = False,
        act_on_cat: bool = False,
        multiscale: bool = True,
    ):
        super().__init__()
        assert len(encoder_channels) == len(encoder_strides)
        assert n_blocks < len(decoder_channels)
        if isinstance(encoder_strides[0], int):
            encoder_strides = [(e, e) for e in encoder_strides]
        else:
            assert all(
                [
                    len(e) == 2 and isinstance(e[0], int) and isinstance(e[1], int)
                    for e in encoder_strides
                ]
            )
        self.use_center_block = center_block
        self.skip_on_finest = skip_on_finest
        self.multiscale = multiscale

        # if norm/act is provided as string, resolve them to the corresponding nn.Module
        if isinstance(norm_layer, str):
            norm_layer = get_norm(f"{norm_layer}_2d", channel_first=True)
        if isinstance(act_layer, str):
            act_layer = get_act(act_layer, inplace=True)
        if isinstance(conv_layer, str):
            conv_layer = get_conv_layer(conv_layer)

        if conv_kwargs is not None:
            conv_layer = partial(conv_layer, **conv_kwargs)

        # coarse to fine
        encoder_channels = encoder_channels[::-1]
        encoder_strides = encoder_strides[::-1]

        # the first value of decoder_channels corresponding to the coarsest feature map's channels
        # if not provided, inherit from encoder
        if decoder_channels[0] is None or decoder_channels[0] <= 0:
            # coarsest feature map
            decoder_channels[0] = encoder_channels[0]

        # input channels to the UNet Decoder
        # coarse to fine
        in_channels = decoder_channels[:n_blocks]

        # skip channels to the UNet Decoder, length = n_blocks
        # coarse to fine
        # example: encoder_channels = (512, 256, 128, 64, 32) of strides (32, 16, 8, 4, 2)
        # typically with uniform stride of 2, n_blocks = 6 will return a feature map of stride 32 / (2**6) = 0.5
        # that is, 2x resolution (4x area) compared to original input image
        # skip_channels will be (256, 128, 64, 32, 0, 0)
        # where 0 means "no skip connection available"
        if skip_on_finest:
            skip_channels = list(encoder_channels[1:])
        else:
            # don't make skip connection from original resolution image (typically raw pixel intensity)
            skip_channels = list(encoder_channels[1:-1])
        skip_channels.extend([0 for _ in range(max(0, n_blocks - len(skip_channels)))])
        skip_channels = skip_channels[:n_blocks]
        # coarse to fine, start from the second coarsest feature map
        out_channels = list(decoder_channels[1 : n_blocks + 1])
        upscale_factors = [
            (cur[0] // prev[0], cur[1] // prev[1])
            for cur, prev in zip(encoder_strides[:-1], encoder_strides[1:])
        ][:n_blocks]
        upscale_factors.extend(
            [(2, 2) for _ in range(max(0, n_blocks - len(upscale_factors)))]
        )

        assert (
            len(in_channels)
            == len(skip_channels)
            == len(out_channels)
            == len(upscale_factors)
            == n_blocks
        ), str(
            [
                len(in_channels),
                len(skip_channels),
                len(out_channels),
                len(upscale_factors),
                n_blocks,
            ]
        )

        if center_block:
            self.center = UnetCenterBlock(
                encoder_channels[0],
                decoder_channels[0],
                norm_layer=norm_layer,
                act_layer=act_layer,
                conv_layer=conv_layer,
                norm_after_conv=norm_after_conv,
            )
        else:
            assert encoder_channels[0] == decoder_channels[0]
            self.center = nn.Identity()

        # build blocks, forward upsample_type etc.
        self.blocks = nn.ModuleList()
        # print('DEBUG UNET DECODER:', in_channels, skip_channels, out_channels, upscale_factors, encoder_strides)
        for in_ch, skip_ch, out_ch, upscale_factor in zip(
            in_channels, skip_channels, out_channels, upscale_factors
        ):
            block = UnetDecoderBlock(
                in_ch,
                skip_ch,
                out_ch,
                attn_type=attn_type,
                upscale_type=upscale_type,
                upscale_factor=upscale_factor,
                norm_layer=norm_layer,
                act_layer=act_layer,
                conv_layer=conv_layer,
                interpolation_mode=interpolation_mode,
                interpolation_antialias=interpolation_antialias,
                shuffle_blur=shuffle_blur,  # pixelshuffle specific
                shuffle_blur_ksize=shuffle_blur_ksize,  # pixelshuffle specific
                shuffle_post_conv=shuffle_post_conv,  # pixelshuffle specific
                use_residual=use_residual,
                norm_after_conv=norm_after_conv,
                norm_on_skip=norm_on_skip,
                norm_on_cat=norm_on_cat,
                act_on_cat=act_on_cat,
            )
            self.blocks.append(block)

        # fine to coarse
        self._output_channels = out_channels[::-1] + [decoder_channels[0]]

    @property
    def output_channels(self):
        return self._output_channels

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # coarse to fine
        features = features[::-1]
        spatial_shapes = [feature.shape[2:] for feature in features]

        head = features[0]
        skip_connections = features[1:] if self.skip_on_finest else features[1:-1]

        x = self.center(head)

        outputs = [x]
        for i, decoder_block in enumerate(self.blocks):
            height, width = (
                spatial_shapes[i + 1]
                if i + 1 < len(spatial_shapes)
                else (2 * height, 2 * width)
            )
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, height, width, skip_connection=skip_connection)
            outputs.append(x)
        if self.multiscale:
            return outputs[::-1]  # fine to coarse
        else:
            return x
