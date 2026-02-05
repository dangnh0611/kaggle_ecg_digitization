# @TODO - refactor all
# naming should be ChannelMaskedBiKLDivLoss, ChannelMaskedJSDLoss
# which support arbitary number of dimensions and mask
# just accept a mask which is broadcastable to both target and input

import logging

import torch
from torch import nn
from torch.nn import functional as F
import math

__all__ = [
    "ChannelMaskedBiKLDivLoss2d",
    "ChannelMaskedJSDLoss2d",
    "ChannelMaskedBCEWithLogitsLoss2d",
]

logger = logging.getLogger(__name__)


class ChannelMaskedBiKLDivLoss2d(nn.modules.loss._Loss):

    def __init__(self, reduction="batchmean", dim=(2, 3), weights=[0.5, 0.5]):
        super().__init__()
        self.reduction = reduction
        assert len(dim) >= 1
        self.dim = dim
        self.reduce_dim = dim[0] if len(dim) == 1 else -1
        self.forward_w, self.backward_w = weights

    def forward(self, input, target, mask=None):
        """Args:
        input: raw logits (before softmax) of shape (B, C, H, W)
        target: target heatmap of shape (B, C, H, W)
        mask: (B, C) where True -> keep, False -> ignore
        """
        if mask is not None:
            input = torch.masked_fill(input, ~mask[..., None, None], 1.0)
            target = torch.masked_fill(target, ~mask[..., None, None], 1.0)
        if self.reduction == "masked_sum":
            if mask is not None:
                mask_sum = mask.sum()
            else:
                mask_sum = math.prod(target.shape[d % target.ndim] for d in self.dim)

        # @TODO - more robust tensor channel order
        # current impl is WRONG :) which only support consecutive dim, e.g (2, 3), not (1, 3)
        if len(self.dim) > 1:
            input = torch.flatten(input, *self.dim)
            target = torch.flatten(target, *self.dim)

        # normalize target to a probability distribution
        target = target / target.sum(dim=self.reduce_dim, keepdim=True)
        input = F.log_softmax(input, dim=self.reduce_dim)

        kl_reduction = "sum" if self.reduction == "masked_sum" else self.reduction

        loss = 0
        if self.forward_w > 0:
            loss += self.forward_w * F.kl_div(
                input, target, reduction=kl_reduction, log_target=False
            )
        if self.backward_w > 0:
            epsilon = 1e-15
            target = torch.clamp(target, min=epsilon, max=1 - epsilon)
            loss += self.backward_w * F.kl_div(
                target.log(), input, reduction=kl_reduction, log_target=True
            )
        if self.reduction == "masked_sum":
            # prevent divide by 0
            if mask_sum > 0:
                return loss / mask_sum
            else:
                # expect to be 0 as well
                return loss
        else:
            return loss


class ChannelMaskedJSDLoss2d(nn.modules.loss._Loss):
    """Jensen Shannon Divergence Loss for 2D heatmap (B, C, H, W)"""

    def __init__(self, reduction="batchmean", dim=(2, 3)):
        super().__init__()
        self.reduction = reduction
        logger.info("MaskedJSDLoss2d with reduction=%s", reduction)
        assert len(dim) >= 1
        self.dim = dim
        self.reduce_dim = dim[0] if len(dim) == 1 else -1
        self.kl = nn.KLDivLoss(
            reduction="sum" if reduction == "masked_sum" else reduction, log_target=True
        )

    def forward(self, input, target, mask=None):
        """Args:
        input: raw logits (before softmax) of shape (B, C, H, W)
        target: target heatmap of shape (B, C, H, W)
        mask: (B, C) where True -> keep, False -> ignore
        """
        B, C, H, W = target.shape
        if mask is not None:
            input = torch.masked_fill(input, ~mask[..., None, None], 1.0)
            target = torch.masked_fill(target, ~mask[..., None, None], 1.0)

        if self.reduction == "masked_sum":
            if mask is not None:
                mask_sum = mask.sum()
            else:
                mask_sum = math.prod(target.shape[d % target.ndim] for d in self.dim)

        if len(self.dim) > 1:
            input = torch.flatten(input, *self.dim)
            target = torch.flatten(target, *self.dim)

        input = F.log_softmax(input, dim=self.reduce_dim)

        # normalize target to a probability distribution
        target = target / target.sum(dim=self.reduce_dim, keepdim=True)
        epsilon = 1e-15
        target = torch.clamp(target, min=epsilon, max=1 - epsilon)
        M = ((input.exp() + target) * 0.5).log()
        target = target.log()
        jsd_loss = 0.5 * (self.kl(M, input) + self.kl(M, target))

        if self.reduction == "masked_sum":
            # prevent divide by 0
            if mask_sum > 0:
                return jsd_loss / mask_sum
            else:
                # expect to be 0 as well
                return jsd_loss
        else:
            return jsd_loss


class ChannelMaskedBCEWithLogitsLoss2d(nn.Module):

    def __init__(self, weight=None, pos_weight=None) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            weight=weight, reduction="none", pos_weight=pos_weight
        )

    def forward(self, input, target, mask):
        """
        Args:
            input: (B, C, H, W)
            target: (B, C, H, W)
            mask: (B, C), True -> keep, False -> ignore
        """
        loss = self.bce(input, target)
        loss = loss * mask[..., None, None]
        loss = loss.sum()
        mask_sum = mask.sum()
        if mask_sum > 0:
            return loss / mask_sum
        else:
            return loss
