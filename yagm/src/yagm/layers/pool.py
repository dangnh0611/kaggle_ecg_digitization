import math
from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init
from torchvision.ops import StochasticDepth
from yagm.layers.common import MaskedSoftmax1d

from timm.layers.config import use_fused_attn

__all__ = [
    "get_global_pool",
    # pool 2d
    "GEMPooling2d",
    # global pool 2d
    "GlobalGEMPooling2d",
    # pool 1d
    "GEMPooling1d",
    # global pool 1d
    "GlobalGEMPooling1d",
    # global masked pool 1d
    "GlobalMaskedAvgPooling1d",
    "GlobalMaskedGEMPooling1d",
    "GlobalMaskedMaxPooling1d",
    "ChannelLastGlobalMaskedMaxPooling1d",
    "GlobalMaskedAttentionPooling1d",
    "GlobalMaskedConcatAttentionPooling1d",
]


class GEMPooling2d(nn.Module):
    """Generalized Mean Pooling 2D
    Ref: https://amaarora.github.io/posts/2020-08-30-gempool.html
    """

    def __init__(self, ksize=(2, 2), p=3, eps=1e-6):
        super().__init__()
        self.ksize = (ksize, ksize) if isinstance(ksize, int) else ksize
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (N, C, H, W)
        Returns:
            (N, C, H2, W2) where H2, W2 is based on ksize, usually H2 = H // ksize
        """
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), self.ksize).pow(
            1.0 / self.p
        )


class GlobalGEMPooling2d(nn.Module):
    """Generalized Mean Pooling 2D
    Ref: https://amaarora.github.io/posts/2020-08-30-gempool.html
    """

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def gem(self, x, p=3, eps=1e-6):
        return

    def forward(self, x):
        """
        Args:
            x: (N, C, H, W)
        Returns:
            (N, C)
        """
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class GEMPooling1d(nn.Module):
    """Generalized Mean Pooling 1D
    Ref: https://www.kaggle.com/code/scaomath/g2net-1d-cnn-gem-pool-pytorch-train-inference
    """

    def __init__(self, ksize=2, p=3, eps=1e-6):
        super().__init__()
        self.ksize = ksize
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (N, C, L)
        Returns:
            (N, C, L2) where L2 is based on ksize, usually L2 = L // ksize
        """
        return F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), self.ksize).pow(
            1.0 / self.p
        )


class GlobalGEMPooling1d(nn.Module):
    """Generalized Mean Pooling 1D
    Ref: https://www.kaggle.com/code/scaomath/g2net-1d-cnn-gem-pool-pytorch-train-inference
    """

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (N, C, L)
        Returns:
            (N, C)
        """
        return F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), x.size(-1)).pow(
            1.0 / self.p
        )


class GlobalMaskedAvgPooling1d(nn.Module):
    """Global Masked Average Pooling 1D
    Ref: https://github.com/amedprof/Feedback-Prize--English-Language-Learning/blob/main/src/model_zoo/pooling.py
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, C, L)
            mask: (N, L) where pad=False, nonpad=True
        Returns:
            (N, C)
        """
        if mask is None:
            return F.adaptive_avg_pool1d(x, 1)
        padding_mask_expanded = mask.unsqueeze(1).expand(x.size())  # NCL
        sum_embeddings = torch.sum(x * padding_mask_expanded, -1)  # NCL -> NC
        # @TODO (dangnh): optimize
        sum_mask = padding_mask_expanded.sum(-1)  # NCT -> NC
        # sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask  # NC
        return mean_embeddings


class GlobalMaskedGEMPooling1d(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self._avg_pool = GlobalMaskedAvgPooling1d()

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, C, L)
            mask: (N, L) where pad=False, nonpad=True
        Returns:
            (N, C)
        """
        x = x.clamp(min=self.eps).pow(self.p)
        x = self._avg_pool(x, mask)
        x = x.pow(1.0 / self.p)
        return x


class GlobalMaskedMaxPooling1d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, C, L)
            mask: (N, L) where pad=False, nonpad=True
        Returns:
            (N, C)
        """
        if mask is None:
            return F.adaptive_max_pool1d(x, 1)
        padding_mask_expanded = (
            torch.logical_not(mask).unsqueeze(1).expand(x.size())
        )  # NCL
        hidden_state_copy = x.clone()
        hidden_state_copy.masked_fill(padding_mask_expanded, float("-inf"))
        return torch.max(hidden_state_copy, -1)[0]  # NCL -> NC


class ChannelLastGlobalMaskedMaxPooling1d(GlobalMaskedMaxPooling1d):

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, L, C)
            mask: (N, L) where pad=False, nonpad=True
        Returns:
            (N, C)
        """
        x = x.permute(0, 2, 1)  # NLC -> NCL
        return super().forward(x, mask)


class GlobalMaskedAttentionPooling1d(nn.Module):
    """
    Global Masked Attention Pooling 1D
    Paper: https://arxiv.org/pdf/2008.01077v1.pdf
    Ref: https://github.com/daniel-code/TubeViT/blob/main/tubevit/model.py
    """

    def __init__(self, input_dim, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        self.proj = nn.Linear(input_dim, 1)
        self.masked_softmax = MaskedSoftmax1d(-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, C, L) if `channel_first` else (N, L, C) (default)
            mask: (N, L) where pad=False, nonpad=True
        Returns:
            (N, C)
        """
        if self.channel_first:
            # NCL -> NLC
            x = x.permute(0, 2, 1)
        _weight_logit = self.proj(x).squeeze(dim=-1)
        att_w = self.masked_softmax(_weight_logit, mask).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
        return x


class GlobalMaskedConcatAttentionPooling1d(nn.Module):
    """
    Concat of (Attention Pooling + CLS) Pooling
    """

    def __init__(self, input_dim, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        self.proj = nn.Linear(input_dim, 1)
        self.masked_softmax = MaskedSoftmax1d(-1)

    def forward(self, input, mask=None):
        """
        Args:
            x: (N, C, L) if `channel_first` else (N, L, C) (default)
            mask: (N, L) where pad=False, nonpad=True
        Returns:
            (N, 2C)
        """
        if self.channel_first:
            # NCL -> NLC
            x = x.permute(0, 2, 1)
        x = input[:, 1:, ...]
        mask = mask[:, 1:]
        _weight_logit = self.proj(x).squeeze(dim=-1)
        att_w = self.masked_softmax(_weight_logit, mask).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
        x = torch.cat([input[:, 0, :], x], axis=-1)  # N x (2C)
        return x


class GlobalMaskedPooling(nn.Module):
    """
    Generalized masked global pooling (1D / 2D / 3D).

    Supports x of shape:
        1D: (B, C, L)
        2D: (B, C, H, W)
        3D: (B, C, D, H, W)

    mask may be:
        None → normal global pooling (fast path)
        1D: (B, L), (B,1,L), (B,C,L)
        2D: (B, H, W), (B,1,H,W), (B,C,H,W)
        3D: (B, D,H,W), (B,1,D,H,W), (B,C,D,H,W)

    Parameters
    ----------
    mode : {"avg", "sum", "max"}
    eps : float
        Avoid divide-by-zero in avg.
    keepdim : bool
        Output shape keeps spatial dims = 1.
    resize_mask : bool
        Resize mask to x's size if mismatched.
    mask_interp_mode : str
        "nearest", "linear", "bilinear", "trilinear", etc.
    align_corners : bool or None
    mask_bin_threshold : float or None
        If float → mask = (mask >= threshold).
    """

    def __init__(
        self,
        mode: str = "avg",
        eps: float = 1e-6,
        keepdim: bool = False,
        resize_mask: bool = False,
        mask_interp_mode: str = "nearest",
        align_corners: bool | None = None,
        mask_bin_threshold: float | None = None,
    ):
        super().__init__()
        mode = mode.replace("masked_", "")
        assert mode in {"avg", "sum", "max"}
        self.mode = mode
        self.eps = eps
        self.keepdim = keepdim
        self.resize_mask = resize_mask
        self.mask_interp_mode = mask_interp_mode
        self.align_corners = align_corners
        self.mask_bin_threshold = mask_bin_threshold

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Perform pooled = masked_pool(x, mask)
        or normal global pooling if mask=None.
        """
        if x.dim() not in (3, 4, 5):
            raise ValueError(f"x must be 3D/4D/5D, got {x.shape}")

        B, C = x.shape[:2]
        spatial_size = x.shape[2:]
        spatial_dims = tuple(range(2, x.dim()))  # e.g. (2,) or (2,3) or (2,3,4)

        # -------------------------------------------------------
        # FAST PATH: No mask → standard global pooling
        # -------------------------------------------------------
        if mask is None:
            if self.mode == "sum":
                pooled = x.sum(dim=spatial_dims)

            elif self.mode == "avg":
                pooled = x.mean(dim=spatial_dims)

            else:  # max
                pooled = x.amax(dim=spatial_dims)

            if self.keepdim:
                pooled = pooled.view(B, C, *([1] * (x.dim() - 2)))
            return pooled

        # -------------------------------------------------------
        # 1) Normalize mask dimensions
        # -------------------------------------------------------
        if mask.dim() == x.dim() - 1:
            mask = mask.unsqueeze(1)  # (B,1,*spatial)
        elif mask.dim() == x.dim():
            if mask.shape[1] != 1:
                # collapse channels → any >0 is valid
                mask = (mask > 0).float().amax(dim=1, keepdim=True)
        else:
            raise ValueError(
                f"mask must be None or match spatial dims, got {mask.shape}"
            )

        mask = mask.to(dtype=x.dtype)
        mask_spatial = mask.shape[2:]

        # -------------------------------------------------------
        # 2) Resize mask if needed
        # -------------------------------------------------------
        if mask_spatial != spatial_size:
            if not self.resize_mask:
                raise ValueError(
                    f"Mask size {mask_spatial} != input {spatial_size}, "
                    "set resize_mask=True to enable resizing."
                )

            if self.mask_interp_mode == "nearest":
                mask = F.interpolate(mask, size=spatial_size, mode="nearest")
            else:
                mask = F.interpolate(
                    mask,
                    size=spatial_size,
                    mode=self.mask_interp_mode,
                    align_corners=self.align_corners,
                )

        # -------------------------------------------------------
        # 3) Optional binarization
        # -------------------------------------------------------
        if self.mask_bin_threshold is not None:
            mask = (mask >= self.mask_bin_threshold).to(x.dtype)

        # -------------------------------------------------------
        # 4) Broadcast over channels
        # -------------------------------------------------------
        mask = mask.expand(B, C, *spatial_size)

        # -------------------------------------------------------
        # 5) Masked pooling
        # -------------------------------------------------------
        if self.mode in {"avg", "sum"}:
            x_weight = x * mask
            pooled = x_weight.sum(dim=spatial_dims)

            if self.mode == "avg":
                denom = mask.sum(dim=spatial_dims)
                pooled = pooled / (denom + self.eps)

        else:  # max
            very_neg = torch.finfo(x.dtype).min
            x_masked = torch.where(mask > 0, x, torch.full_like(x, very_neg))
            pooled = x_masked.amax(dim=spatial_dims)

        if self.keepdim:
            pooled = pooled.view(B, C, *([1] * (x.dim() - 2)))

        return pooled

    def __repr__(self):
        args = [f"mode={self.mode!r}"]

        if self.eps != 1e-6:
            args.append(f"eps={self.eps}")

        if self.keepdim:
            args.append(f"keepdim={self.keepdim}")

        if self.resize_mask:
            args.append(f"resize_mask={self.resize_mask}")

        if self.mask_interp_mode != "nearest":
            args.append(f"mask_interp_mode={self.mask_interp_mode!r}")

        if self.align_corners is not None:
            args.append(f"align_corners={self.align_corners}")

        if self.mask_bin_threshold is not None:
            args.append(f"mask_bin_threshold={self.mask_bin_threshold}")

        arg_str = ", ".join(args)
        return f"{self.__class__.__name__}({arg_str})"


"""
Copied from timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/other_pool.py

Non-Local Attention Pooling Layers

A collection of global pooling layers that go beyond simple avg/max pooling.

LSEPool - LogSumExp pooling, a smooth approximation between avg and max pooling
SimPool - Attention-based pooling from 'Keep It SimPool' (ICCV 2023)

Based on implementations from:
* LSE Pooling: custom implementation by Bill Psomas
* SimPool: https://arxiv.org/abs/2309.06891 - 'Keep It SimPool: Who Said Supervised Transformers
    Suffer from Attention Deficit?' by Bill Psomas et al.

Hacked together by / Copyright 2024 Ross Wightman, original code by Bill Psomas
"""


class LsePlus2d(nn.Module):
    """LogSumExp (LSE) Pooling for 2D inputs.

    A smooth approximation to max pooling that provides a learnable interpolation between
    average and max pooling. When r is large, LSE approaches max pooling; when r is small,
    it approaches average pooling.

    Implements: (1/r) * log((1/n) * sum(exp(r * (x - x_max)))) + x_max

    The x_max subtraction provides numerical stability.
    """

    def __init__(
        self,
        r: float = 10.0,
        r_learnable: bool = True,
        flatten: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Args:
            r: Initial value of the pooling parameter. Higher = closer to max pooling.
            r_learnable: If True, r is a learnable parameter.
            flatten: If True, flatten spatial dims in output.
        """
        super().__init__()
        if r_learnable:
            self.r = nn.Parameter(torch.tensor(r, device=device, dtype=dtype))
        else:
            self.register_buffer("r", torch.tensor(r, device=device, dtype=dtype))
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = F.adaptive_max_pool2d(x, 1)
        exp_x = torch.exp(self.r * (x - x_max))
        sum_exp = exp_x.mean(dim=(2, 3), keepdim=True)
        out = x_max + (1.0 / self.r) * torch.log(sum_exp)
        if self.flatten:
            out = out.flatten(1)
        return out


class LsePlus1d(nn.Module):
    """LogSumExp (LSE) Pooling for sequence (NLC) inputs.

    A smooth approximation to max pooling that provides a learnable interpolation between
    average and max pooling. When r is large, LSE approaches max pooling; when r is small,
    it approaches average pooling.
    """

    def __init__(
        self,
        r: float = 10.0,
        r_learnable: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Args:
            r: Initial value of the pooling parameter. Higher = closer to max pooling.
            r_learnable: If True, r is a learnable parameter.
        """
        super().__init__()
        if r_learnable:
            self.r = nn.Parameter(torch.tensor(r, device=device, dtype=dtype))
        else:
            self.register_buffer("r", torch.tensor(r, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        x_max = x.max(dim=1, keepdim=True).values
        exp_x = torch.exp(self.r * (x - x_max))
        sum_exp = exp_x.mean(dim=1, keepdim=True)
        out = x_max + (1.0 / self.r) * torch.log(sum_exp)
        return out.squeeze(1)  # (B, C)


class SimPool2d(nn.Module):
    """SimPool: Simple Attention-Based Pooling for 2D (NCHW) inputs.

    From 'Keep It SimPool: Who Said Supervised Transformers Suffer from Attention Deficit?'
    https://arxiv.org/abs/2309.06891

    Uses GAP as query initialization and applies cross-attention between the GAP query
    and spatial features to produce a weighted pooled representation.
    """

    fused_attn: torch.jit.Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        gamma: Optional[float] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
        device=None,
        dtype=None,
    ):
        """
        Args:
            dim: Input feature dimension (number of channels).
            num_heads: Number of attention heads.
            qkv_bias: If True, add bias to query and key projections.
            qk_norm: If True, apply normalization to queries and keys.
            gamma: If provided, apply power normalization to values with this exponent.
            norm_layer: Normalization layer for patches and optionally qk_norm.
            flatten: If True, flatten output to (B, C).
        """
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.gamma = gamma
        self.fused_attn = use_fused_attn()

        norm_layer = norm_layer or nn.LayerNorm
        self.norm = norm_layer(dim, **dd)
        self.q = nn.Linear(dim, dim, bias=qkv_bias, **dd)
        self.k = nn.Linear(dim, dim, bias=qkv_bias, **dd)
        if qk_norm:
            self.q_norm = norm_layer(self.head_dim, **dd)
            self.k_norm = norm_layer(self.head_dim, **dd)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # Reshape to (B, N, C) for attention
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # GAP as query initialization
        q = x.mean(dim=1, keepdim=True)  # (B, 1, C)

        # Normalize patches for keys and values
        x_norm = self.norm(x)

        # Project query and keys
        q = self.q(q).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = x_norm.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.gamma is not None:
            # Power normalization on values
            v_min = v.amin(dim=-2, keepdim=True)
            v_shifted = v - v_min + 1e-6
            if self.fused_attn:
                attn_out = F.scaled_dot_product_attention(
                    q, k, v_shifted.pow(self.gamma)
                )
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn_out = attn @ v_shifted.pow(self.gamma)
            out = attn_out.pow(1.0 / self.gamma)
        else:
            if self.fused_attn:
                out = F.scaled_dot_product_attention(q, k, v)
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                out = attn @ v

        # (B, num_heads, 1, head_dim) -> (B, C) or (B, C)
        out = out.transpose(1, 2).reshape(B, C)
        return out


class SimPool1d(nn.Module):
    """SimPool: Simple Attention-Based Pooling for sequence (NLC) inputs.

    From 'Keep It SimPool: Who Said Supervised Transformers Suffer from Attention Deficit?'
    https://arxiv.org/abs/2309.06891

    Uses GAP as query initialization and applies cross-attention between the GAP query
    and sequence tokens to produce a weighted pooled representation.
    """

    fused_attn: torch.jit.Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        gamma: Optional[float] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
        device=None,
        dtype=None,
    ):
        """
        Args:
            dim: Input feature dimension.
            num_heads: Number of attention heads.
            qkv_bias: If True, add bias to query and key projections.
            qk_norm: If True, apply normalization to queries and keys.
            gamma: If provided, apply power normalization to values with this exponent.
            norm_layer: Normalization layer for tokens and optionally qk_norm.
        """
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.gamma = gamma
        self.fused_attn = use_fused_attn()

        norm_layer = norm_layer or nn.LayerNorm
        self.norm = norm_layer(dim, **dd)
        self.q = nn.Linear(dim, dim, bias=qkv_bias, **dd)
        self.k = nn.Linear(dim, dim, bias=qkv_bias, **dd)
        if qk_norm:
            self.q_norm = norm_layer(self.head_dim, **dd)
            self.k_norm = norm_layer(self.head_dim, **dd)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # GAP as query initialization
        q = x.mean(dim=1, keepdim=True)  # (B, 1, C)

        # Normalize tokens for keys and values
        x_norm = self.norm(x)

        # Project query and keys
        q = self.q(q).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = x_norm.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.gamma is not None:
            # Power normalization on values
            v_min = v.amin(dim=-2, keepdim=True)
            v_shifted = v - v_min + 1e-6
            if self.fused_attn:
                attn_out = F.scaled_dot_product_attention(
                    q, k, v_shifted.pow(self.gamma)
                )
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn_out = attn @ v_shifted.pow(self.gamma)
            out = attn_out.pow(1.0 / self.gamma)
        else:
            if self.fused_attn:
                out = F.scaled_dot_product_attention(q, k, v)
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                out = attn @ v

        # (B, num_heads, 1, head_dim) -> (B, C)
        out = out.transpose(1, 2).reshape(B, C)
        return out


def normalize_and_resize_mask(
    mask: torch.Tensor,
    x: torch.Tensor,
    *,
    resize_mask: bool,
    interp_mode: str = "nearest",
    align_corners: bool | None = None,
    bin_threshold: float | None = None,
):
    """Normalizes and resizes a spatial mask to match the input tensor's dimensions.

    This utility ensures that a mask (which may be provided in various formats, such
    as a 1D sequence mask, a 2D segmentation mask, or a multi-channel mask) is
    correctly broadcastable to the spatial dimensions of the input tensor `x`.

    Args:
        mask (torch.Tensor): The input mask. Expected shapes:
            - (B, *spatial)
            - (B, 1, *spatial)
            - (B, C, *spatial) (will be reduced to binary)
        x (torch.Tensor): The reference input tensor with shape (B, C, *spatial).
        resize_mask (bool): If True, resizes the mask to match `x`. If False,
            raises ValueError on mismatch.
        interp_mode (str): Interpolation mode for `F.interpolate`. Defaults to "nearest".
        align_corners (bool, optional): `align_corners` parameter for interpolation.
        bin_threshold (float, optional): If provided, binarizes the mask at this
            threshold after resizing.

    Returns:
        torch.Tensor: Normalized mask of shape (B, 1, *spatial) and same dtype as `x`.

    Raises:
        ValueError: If the mask dimensions are incompatible with `x` or if
            `resize_mask` is False but shapes do not match.
    """
    # 1. Align Dimensions
    if mask.dim() == x.dim() - 1:
        mask = mask.unsqueeze(1)
    elif mask.dim() == x.dim():
        if mask.shape[1] != 1:
            # Flatten channel-wise masks to spatial binary mask
            mask = (mask > 0).float().amax(dim=1, keepdim=True)
    else:
        raise ValueError(f"Invalid mask shape {mask.shape} for x {x.shape}")

    mask = mask.to(dtype=x.dtype)

    # 2. Resize if necessary
    spatial_size = x.shape[2:]
    if mask.shape[2:] != spatial_size:
        if not resize_mask:
            raise ValueError(f"Mask shape {mask.shape[2:]} != input {spatial_size}")

        mask = F.interpolate(
            mask,
            size=spatial_size,
            mode=interp_mode,
            align_corners=align_corners if interp_mode != "nearest" else None,
        )

    # 3. Binarize
    if bin_threshold is not None:
        mask = (mask >= bin_threshold).to(dtype=x.dtype)

    return mask


class MaskedLsePlus(nn.Module):
    """Masked LogSumExp (LSE) Pooling for 1D, 2D, or 3D inputs.

    LSE provides a smooth, learnable interpolation between average pooling and
    max pooling. As $r \to \infty$, it approaches max pooling; as $r \to 0$,
    it approaches average pooling.

    The masked implementation handles numerical stability by using a finite
    large negative constant for background regions, preventing the $x_{max}$
    term from being corrupted by padding or ignored pixels.

    Formula:
    $$y = x_{max} + \frac{1}{r} \log \left( \frac{1}{\sum M} \sum e^{r(x - x_{max})} \cdot M \right)$$
    where $M$ is the binary mask.

    Attributes:
        r (nn.Parameter or torch.Tensor): The pooling sharpness parameter.
        eps (float): Small constant for numerical stability in division and log.
        keepdim (bool): Whether to maintain spatial dimensions (as 1) in output.
        resize_mask (bool): Whether to automatically resize provided masks.
    """

    def __init__(
        self,
        r: float = 10.0,
        r_learnable: bool = True,
        eps: float = 1e-6,
        keepdim: bool = False,
        resize_mask: bool = False,
        mask_interp_mode: str = "nearest",
        mask_bin_threshold: float = 0.5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if r_learnable:
            self.r = nn.Parameter(torch.tensor(r, device=device, dtype=dtype))
        else:
            self.register_buffer("r", torch.tensor(r, device=device, dtype=dtype))

        self.eps = eps
        self.keepdim = keepdim
        self.resize_mask = resize_mask
        self.mask_interp_mode = mask_interp_mode
        self.mask_bin_threshold = mask_bin_threshold

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        if x.dim() not in (3, 4, 5):
            raise ValueError("x must be 1D / 2D / 3D tensor")

        spatial_dims = tuple(range(2, x.dim()))

        if mask is not None:
            mask = normalize_and_resize_mask(
                mask,
                x,
                resize_mask=self.resize_mask,
                interp_mode=self.mask_interp_mode,
                bin_threshold=self.mask_bin_threshold,
            )

        if mask is None:
            # Fast path (Standard LSE)
            x_max = x.amax(dim=spatial_dims, keepdim=True)
            exp_x = torch.exp(self.r * (x - x_max))
            mean_exp = exp_x.mean(dim=spatial_dims, keepdim=True)
        else:
            # --- Robust Masked Logic ---
            # 1. Mask Input: Use -1e9 for background (Safe negative).
            # Do NOT use 0 (0 > negative features). Do NOT use -inf (NaNs in subtraction).
            very_neg = -1e9
            mask_bool = mask > 0

            x_masked = torch.where(
                mask_bool, x, torch.tensor(very_neg, dtype=x.dtype, device=x.device)
            )

            # 2. Compute Max (Valid pixels only)
            x_max = x_masked.amax(dim=spatial_dims, keepdim=True)

            # Safety: If sample is empty, x_max is -1e9. Clamp to 0 to prevent NaN propagation.
            x_max = torch.where(x_max > (very_neg + 1), x_max, torch.zeros_like(x_max))

            # 3. Compute Exp
            # (x - x_max) for valid pixels.
            # (very_neg - x_max) -> large neg -> exp -> 0.
            exp_x = torch.exp(self.r * (x_masked - x_max))

            # Explicitly zero out background for float safety
            exp_x = exp_x * mask

            denom = mask.sum(dim=spatial_dims, keepdim=True)
            mean_exp = exp_x.sum(dim=spatial_dims, keepdim=True) / (denom + self.eps)

        out = x_max + (1.0 / self.r) * torch.log(mean_exp + self.eps)

        if not self.keepdim:
            out = out.flatten(1)

        return out


class MaskedSimPool2d(nn.Module):
    """Masked Simple Attention-Based Pooling (SimPool) for 2D inputs.

    SimPool utilizes a cross-attention mechanism where the "Query" is initialized
    via Global Average Pooling (GAP) and refined by attending to spatial features.
    This allows the model to learn which spatial regions are most relevant for
    the pooled representation.

    The masked version ensures that:
    1. The GAP query initialization ignores masked regions.
    2. Attention is strictly constrained to valid (foreground) pixels.
    3. Power normalization (gamma) is calculated only over foreground statistics
       to prevent background noise from shifting feature distributions.

    Reference:
        "Keep It SimPool: Who Said Supervised Transformers Suffer from Attention Deficit?"
        https://arxiv.org/abs/2309.06891

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, adds learnable bias to query/key projections.
        qk_norm (bool): If True, applies normalization to queries and keys.
        gamma (float, optional): Exponent for power normalization. If provided,
            applies $v = (v - v_{min})^{\gamma}$ before attention.
        resize_mask (bool): Whether to resize input masks to match feature maps.
        mask_bin_threshold (float): Threshold to binarize soft masks."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        gamma: float | None = None,
        norm_layer: type[nn.Module] | None = None,
        resize_mask: bool = False,
        mask_interp_mode: str = "nearest",
        mask_bin_threshold: float = 0.5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.gamma = gamma
        self.resize_mask = resize_mask
        self.mask_interp_mode = mask_interp_mode
        self.mask_bin_threshold = mask_bin_threshold

        self.fused_attn = hasattr(F, "scaled_dot_product_attention")

        norm_layer = norm_layer or nn.LayerNorm
        self.norm = norm_layer(dim, **dd)
        self.q = nn.Linear(dim, dim, bias=qkv_bias, **dd)
        self.k = nn.Linear(dim, dim, bias=qkv_bias, **dd)
        if qk_norm:
            self.q_norm = norm_layer(self.head_dim, **dd)
            self.k_norm = norm_layer(self.head_dim, **dd)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, C, H, W = x.shape
        N = H * W

        # --- Mask Preparation ---
        if mask is not None:
            mask = normalize_and_resize_mask(
                mask,
                x,
                resize_mask=self.resize_mask,
                interp_mode=self.mask_interp_mode,
                bin_threshold=self.mask_bin_threshold,
            )
            # mask: (B, 1, H, W) -> (B, N, 1)
            # mask contains 1.0 (Keep) and 0.0 (Ignore)
            mask_flat = mask.flatten(2).transpose(1, 2)
        else:
            mask_flat = None

        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # --- 1. Masked Query Initialization ---
        if mask_flat is not None:
            # Weighted average based on mask
            denom = mask_flat.sum(dim=1, keepdim=True).clamp_min(1.0)
            q_pooled = (x * mask_flat).sum(dim=1, keepdim=True) / denom
        else:
            q_pooled = x.mean(dim=1, keepdim=True)

        x_norm = self.norm(x)

        # Projections
        q = (
            self.q(q_pooled)
            .reshape(B, 1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = self.k(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = x_norm.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        # --- 2. Construct Attention Mask ---
        attn_mask = None
        if mask_flat is not None:
            # PyTorch SDPA Boolean Mask:
            # True = Attend (Keep)
            # False = Ignore (Mask out)
            # mask_flat is 1.0 for Keep, 0.0 for Ignore.
            # Shape: (B, 1, 1, N) for broadcast
            attn_mask = (mask_flat > 0.5).squeeze(-1).unsqueeze(1).unsqueeze(1)

        # --- 3. Gamma Power Norm ---
        if self.gamma is not None:
            if mask_flat is not None:
                # Calculate Min of FOREGROUND (valid pixels)
                # Set background to +inf so min() ignores it
                v_for_min = torch.where(
                    mask_flat.unsqueeze(1) > 0,
                    v,
                    torch.tensor(float("inf"), dtype=v.dtype, device=v.device),
                )
                v_min = v_for_min.amin(dim=-2, keepdim=True)
                v_min = torch.where(torch.isinf(v_min), torch.zeros_like(v_min), v_min)
            else:
                v_min = v.amin(dim=-2, keepdim=True)

            v_shifted = v - v_min + 1e-6

            # CRITICAL: Zero out background before .pow()
            # If we don't, background becomes (v_bg - v_min_fg) which can be negative.
            # Negative.pow(float) = NaN.
            if mask_flat is not None:
                # Fill with 0.0 where mask is 0 (Ignore)
                v_shifted = v_shifted.masked_fill(mask_flat.unsqueeze(1) == 0, 0.0)

            v_in = v_shifted.pow(self.gamma)

            if self.fused_attn:
                out = F.scaled_dot_product_attention(q, k, v_in, attn_mask=attn_mask)
            else:
                # Manual fallback
                attn = (q * self.scale) @ k.transpose(-2, -1)
                if attn_mask is not None:
                    # masked_fill fills where mask is True.
                    # We want to fill where attn_mask is False (Ignore).
                    attn = attn.masked_fill(~attn_mask, -float("inf"))
                attn = attn.softmax(dim=-1)
                out = attn @ v_in

            out = out.pow(1.0 / self.gamma)

        else:
            # No Gamma
            if self.fused_attn:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                if attn_mask is not None:
                    attn = attn.masked_fill(~attn_mask, -float("inf"))
                attn = attn.softmax(dim=-1)
                out = attn @ v

        out = out.transpose(1, 2).reshape(B, C)
        return out


def get_global_pool(pool_type, channel_first=True, **kwargs):
    if pool_type == "avg_1d":
        if channel_first:
            return partial(nn.AdaptiveAvgPool1d, 1)
        raise NotImplementedError
    elif pool_type == "max_1d":
        if channel_first:
            return partial(nn.AdaptiveMaxPool1d, 1)
        raise NotImplementedError
    elif pool_type == "masked_avg_1d":
        if channel_first:
            return GlobalMaskedAvgPooling1d
        raise NotImplementedError
    elif pool_type == "masked_max_1d":
        if channel_first:
            return GlobalMaskedMaxPooling1d
        else:
            return ChannelLastGlobalMaskedMaxPooling1d
    elif pool_type == "masked_gem_1d":
        if channel_first:
            return partial(
                GlobalGEMPooling1d, p=kwargs.get("p", 3), eps=kwargs.get("eps", 1e-6)
            )
        raise NotImplementedError
    elif pool_type == "masked_attn_1d":
        return partial(
            GlobalMaskedAttentionPooling1d,
            input_dim=kwargs["input_dim"],
            channel_first=channel_first,
        )
    elif pool_type == "masked_concat_attn_1d":
        return partial(
            GlobalMaskedConcatAttentionPooling1d,
            input_dim=kwargs["input_dim"],
            channel_first=channel_first,
        )
    else:
        raise NotImplementedError
