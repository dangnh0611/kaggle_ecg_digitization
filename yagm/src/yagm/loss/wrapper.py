import torch
import torch.nn as nn


class PerChannelWeightedLossWrapper(nn.Module):
    """
    Wrap a base loss (inner) that returns per-element losses (reduction='none')
    and apply per-channel weights along a chosen dimension.

    Typical usage:
        - input, target: shape (N, C) or (N, C, H, W), etc.
        - weight: shape (C,)
        - dim: which dimension is the "channel" (default: 1)

    The wrapper then:
        1. Multiplies losses by per-channel weights.
        2. Reduces according to `reduction` ("mean" / "sum" / "none").
    """

    def __init__(
        self,
        inner: nn.Module,
        weight: torch.Tensor,
        dim: int = 1,
        normalize: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        assert weight.ndim == 1, f"weight must be 1D, got shape {tuple(weight.shape)}"
        assert reduction in ("mean", "sum", "none")

        self.inner = inner
        self.dim = dim
        self.reduction = reduction

        # Optionally normalize so average weight ≈ 1 (keeps loss scale similar)
        if normalize:
            w = weight / weight.sum() * weight.numel()
        else:
            w = weight

        # Register as buffer so it moves with .to(device) / .cuda() / DDP, etc.
        self.register_buffer("weight", w)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # inner should be reduction='none' so we can apply weights
        loss = self.inner(input, target)

        if loss.ndim == 0:
            # inner loss already reduced; nothing to weight sensibly
            # fall back to just returning it
            return loss

        # Resolve dim (support negative indices)
        dim = self.dim if self.dim >= 0 else loss.ndim + self.dim
        assert 0 <= dim < loss.ndim, f"Invalid dim={self.dim} for loss.ndim={loss.ndim}"

        C = loss.size(dim)
        assert (
            C <= self.weight.numel()
        ), f"Channel size {C} != weight size {self.weight.numel()}"

        # Build a broadcastable weight shape: (1, 1, ..., C, ..., 1)
        # where C is at position `dim`
        shape = [1] * loss.ndim
        shape[dim] = C
        w = self.weight[:C].view(*shape)

        weighted = loss * w

        if self.reduction == "none":
            return weighted
        elif self.reduction == "mean":
            return weighted.mean()
        elif self.reduction == "sum":
            return weighted.sum()
        else:
            # Should never happen because we assert in __init__
            raise RuntimeError(f"Unknown reduction: {self.reduction}")
