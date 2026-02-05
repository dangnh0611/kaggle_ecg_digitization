import torch
import torch.nn as nn


__all__ = [
    "LogCoshLoss",
    "CharbonnierLoss",
    "QuantileLoss",
    "BerHuLoss",
    "CauchyLoss",
    "TukeyLoss",
    "GemanMcClureLoss",
    "WelschLoss",
    "FairLoss",
    "AdaptiveLoss",
    "WeightedMSELoss",
    "WeightedL1Loss",
    "WeightedSmoothL1Loss",
]

EPS = torch.finfo(torch.float16).eps

# =============================================================================
# Custom regression loss functions
# =============================================================================


class LogCoshLoss(nn.Module):
    r"""
    log-cosh loss: L(e) = log(cosh(e))

    Behaves like:
    - ~ 0.5 * e^2 for small e (similar to L2)
    - ~ |e| - log(2) for large e (similar to L1)

    Pros
    ----
    - Smooth everywhere (unlike L1).
    - Less sensitive to outliers than L2, but still penalizes them more
      strongly than Huber with a small delta.
    - Often works well as a drop-in refinement of MSE.

    Cons
    ----
    - Slightly more expensive than pure L1/L2 (cosh and log operations).
    - Hyperparameters are implicit; less direct control over the transition
      between "L1-like" and "L2-like" behavior.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        x = input - target
        loss = torch.log(torch.cosh(x + 1e-12))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CharbonnierLoss(nn.Module):
    r"""
    Charbonnier loss: L(e) = sqrt(e^2 + eps^2)

    A smooth approximation to L1 (a.k.a. "pseudo-Huber" with fixed shape).

    Pros
    ----
    - Differentiable everywhere including at 0.
    - Robust to outliers (less aggressive than pure L2).
    - Very common in vision tasks (optical flow, depth, denoising).

    Cons
    ----
    - Requires choosing epsilon; too small behaves like L1, too large
      gets close to L2.
    - Slightly more expensive than pure L1 (sqrt).
    """

    def __init__(self, epsilon: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, input, target):
        x = input - target
        loss = torch.sqrt(x * x + self.epsilon * self.epsilon)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class QuantileLoss(nn.Module):
    r"""
    Quantile / pinball loss for quantile regression.

    For quantile q in (0,1), error e = target - input:

        L(e) = max(q * e, (q - 1) * e)

    Pros
    ----
    - Directly models conditional quantiles (median, 90th percentile, etc.),
      not just mean.
    - Very useful in forecasting, risk estimation, and when asymmetry is
      fundamental to the problem.

    Cons
    ----
    - Non-symmetric by design; if you just want a central tendency metric,
      MSE/L1 is simpler.
    - Gradient can be less stable if labels are noisy and q is extreme
      (very close to 0 or 1).
    """

    def __init__(self, q: float = 0.5, reduction: str = "mean"):
        super().__init__()
        assert 0.0 < q < 1.0, "q must be in (0, 1)"
        self.q = q
        self.reduction = reduction

    def forward(self, input, target):
        e = target - input
        loss = torch.max(self.q * e, (self.q - 1.0) * e)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BerHuLoss(nn.Module):
    r"""
    Reverse Huber (BerHu) loss.

    One common form:

        if |e| <= c:  L(e) = |e|
        else:         L(e) = (e^2 + c^2) / (2c)

    Pros
    ----
    - L1-like near zero (good robustness) but becomes L2-like for large errors,
      thus **emphasizing big errors**.
    - Useful when you want to focus more on correcting large mistakes.

    Cons
    ----
    - Requires a scale parameter c; if not set well, behavior can be odd.
    - If c is chosen from batch statistics, the loss landscape changes with
      the current mini-batch distribution.
    """

    def __init__(self, c: float | None = None, reduction: str = "mean"):
        super().__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, input, target):
        x = torch.abs(input - target)
        if self.c is None:
            max_x = x.max().detach()
            if max_x == 0:
                loss = x
                if self.reduction == "mean":
                    return loss.mean()
                elif self.reduction == "sum":
                    return loss.sum()
                else:
                    return loss
            c = 0.2 * max_x
        else:
            c = self.c

        small = x <= c
        loss = torch.zeros_like(x)
        loss[small] = x[small]
        loss[~small] = (x[~small] ** 2 + c * c) / (2.0 * c)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CauchyLoss(nn.Module):
    r"""
    Cauchy robust loss:

        L(e) = 0.5 * sigma^2 * log(1 + (e / sigma)^2)

    Pros
    ----
    - Heavy-tailed; aggressively downweights strong outliers.
    - Smooth and differentiable everywhere.

    Cons
    ----
    - Requires choosing sigma (scale); poor choices make it either almost L2
      (too large) or almost L0-like (too small).
    - A bit more expensive than L1/L2.
    """

    def __init__(self, sigma: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, input, target):
        x = input - target
        z = x / self.sigma
        loss = 0.5 * (self.sigma**2) * torch.log1p(z * z)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class TukeyLoss(nn.Module):
    r"""
    Tukey's biweight (bisquare) loss:

        r = e / c
        if |r| < 1:
            L(e) = (c^2 / 6) * [1 - (1 - r^2)^3]
        else:
            L(e) = c^2 / 6

    Pros
    ----
    - Extremely robust: very large errors get **bounded** loss contribution.
    - Great when you expect a significant fraction of completely bad labels
      or measurements that should not influence the fit at all.

    Cons
    ----
    - Non-convex; optimization can be trickier and more sensitive to init.
    - Requires choosing c (scale); not trivial to set universally.
    """

    def __init__(self, c: float = 4.685, reduction: str = "mean"):
        super().__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, input, target):
        e = input - target
        r = e / self.c
        mask = r.abs() < 1
        loss = torch.zeros_like(r)
        loss[mask] = (self.c**2 / 6.0) * (1.0 - (1.0 - r[mask] ** 2) ** 3)
        loss[~mask] = self.c**2 / 6.0

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class GemanMcClureLoss(nn.Module):
    r"""
    Geman–McClure robust loss:

        L(e) = e^2 / (e^2 + sigma^2)

    Pros
    ----
    - Bounded loss; large residuals saturate.
    - Simple and smooth; nice alternative to Cauchy/Charbonnier.

    Cons
    ----
    - Non-convex; may complicate optimization if combined with other non-convex
      components.
    - Requires choosing sigma; sensitive to data scaling.
    """

    def __init__(self, sigma: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, input, target):
        e = input - target
        loss = (e**2) / (e**2 + self.sigma**2)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WelschLoss(nn.Module):
    r"""
    Welsch (Leclerc) loss:

        L(e) = 1 - exp(-(e / sigma)^2)

    Pros
    ----
    - Very strong downweighting of outliers; contributions approach 1 as
      |e| >> sigma.
    - Smooth and easy to implement.

    Cons
    ----
    - Non-convex.
    - Choice of sigma is important; too small -> almost binary penalty,
      too large -> almost quadratic near 0.
    """

    def __init__(self, sigma: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, input, target):
        e = input - target
        loss = 1.0 - torch.exp(-((e / self.sigma) ** 2))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FairLoss(nn.Module):
    r"""
    Fair loss:

        L(e) = c^2 * ( |e|/c - log(1 + |e|/c) )

    Pros
    ----
    - Smooth interpolation between L1 and L2 behaviors.
    - More robust than L2, less harsh than hard-bounded losses (Tukey).

    Cons
    ----
    - Requires setting c; poor scaling can hurt convergence.
    - Slightly more expensive than L1/L2 due to log().
    """

    def __init__(self, c: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, input, target):
        e = torch.abs(input - target)
        x = e / self.c
        loss = self.c**2 * (x - torch.log1p(x))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class AdaptiveLoss(nn.Module):
    r"""
    Barron adaptive robust loss (learnable version by default).

    This implements the robust loss family from:
        "A General and Adaptive Robust Loss Function" - Barron, 2019.

    We use a *learnable* shape parameter alpha and a *learnable* scale
    parameter, so the loss can adapt itself to the noise characteristics
    of the data instead of you manually tuning them.

    For alpha ≠ 0:

        e  = (input - target) / scale
        L(e; alpha, scale) =
            (|alpha - 2| / |alpha|) * [ ((e^2 / |alpha - 2|) + 1)^(alpha/2) - 1 ]

    In practice we:
      - Keep alpha in a safe range (e.g. [-2, 2]).
      - Avoid division by zero by clamping |alpha| and |alpha - 2| to a
        small positive minimum.

    Behavior for different alpha:
      - alpha ≈  2 : ~ L2 (quadratic)
      - alpha ≈  1 : ~ Charbonnier / pseudo-Huber
      - alpha ≈  0 : ~ Cauchy / log loss
      - alpha <  0 : heavy-tailed / Welsch-like

    Parameters
    ----------
    init_alpha : float, default=1.0
        Initial value for alpha (shape parameter). Typical range [-2, 2].
        alpha is learnable by default.

    init_scale : float, default=1.0
        Initial scale for the residuals. Roughly similar to a robust
        estimate of standard deviation. Learnable by default.

    learn_alpha : bool, default=True
        Whether to treat alpha as an nn.Parameter (learned during training).

    learn_scale : bool, default=True
        Whether to treat scale as an nn.Parameter (via log_scale for positivity).

    alpha_range : tuple[float, float] | None, default=(-2.0, 2.0)
        If not None, clamp alpha into this range on each forward pass.

    reduction : {"mean", "sum", "none"}, default="mean"
        Final reduction over elements.

    Pros
    ----
    - Very flexible: can approximate a wide variety of robust losses
      (L2, L1, Cauchy, Charbonnier, Welsch, etc.) via a *single* learnable
      alpha and scale.
    - With learnable parameters, you often need **little to no manual
      tuning** for noise shape/scale.
    - Good default when you don't know the label noise distribution.

    Cons
    ----
    - More complex than basic losses. Adds 1–2 extra learned scalars per
      loss instance (alpha, scale).
    - Slightly more expensive (pow, log1p, etc.).
    - If your model is already unstable / hard to optimize, the extra
      degrees of freedom can complicate training (though in practice it's
      usually fine for well-scaled problems).
    """

    def __init__(
        self,
        init_alpha: float = 1.0,
        init_scale: float = 1.0,
        learn_alpha: bool = True,
        learn_scale: bool = True,
        alpha_range: tuple[float, float] | None = (-2.0, 2.0),
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha_range = alpha_range
        self.reduction = reduction

        # alpha: shape parameter (can be negative, but we keep it bounded)
        alpha_tensor = torch.tensor(float(init_alpha))
        if learn_alpha:
            self.alpha_param = nn.Parameter(alpha_tensor)
        else:
            self.register_buffer("alpha_param", alpha_tensor)

        # scale: must stay positive -> we store log_scale
        log_scale_tensor = torch.log(torch.tensor(float(init_scale)))
        if learn_scale:
            self.log_scale = nn.Parameter(log_scale_tensor)
        else:
            self.register_buffer("log_scale", log_scale_tensor)

    @property
    def alpha(self) -> torch.Tensor:
        a = self.alpha_param
        if self.alpha_range is not None:
            a = torch.clamp(a, self.alpha_range[0], self.alpha_range[1])
        return a

    @property
    def scale(self) -> torch.Tensor:
        # Ensure strictly positive scale
        return self.log_scale.exp().clamp(min=1e-6)

    def forward(self, input, target):
        e = (input - target) / self.scale
        a = self.alpha

        # We need to avoid division by zero / singularities for alpha≈0 and alpha≈2.
        # Use clamped magnitudes in the formula, but keep the original a in exponents.
        eps = 1e-2
        abs_a = torch.clamp(torch.abs(a), min=eps)  # |a|
        abs_a_minus_2 = torch.clamp(torch.abs(a - 2.0), min=eps)  # |a - 2|

        # General Barron loss formula
        loss = (abs_a_minus_2 / abs_a) * (
            (e**2 / abs_a_minus_2 + 1.0) ** (a / 2.0) - 1.0
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WeightedMSELoss(nn.MSELoss):

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, pred, target, weight = None):
        loss = super().forward(pred, target)
        if weight is not None:
            weight = torch.broadcast_to(weight, loss.shape)
            loss = (loss * weight).sum() / (weight.sum() + EPS)
        else:
            loss = loss.mean()
        return loss


class WeightedL1Loss(nn.L1Loss):

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, pred, target, weight = None):
        loss = super().forward(pred, target)
        if weight is not None:
            weight = torch.broadcast_to(weight, loss.shape)
            loss = (loss * weight).sum() / (weight.sum() + EPS)
        else:
            loss = loss.mean()
        return loss


class WeightedSmoothL1Loss(nn.SmoothL1Loss):

    def __init__(self, beta=1.0):
        super().__init__(reduction="none", beta=beta)

    def forward(self, pred, target, weight = None):
        loss = super().forward(pred, target)
        if weight is not None:
            weight = torch.broadcast_to(weight, loss.shape)
            loss = (loss * weight).sum() / (weight.sum() + EPS)
        else:
            # fallback to reduction=mean if weight is not specified
            loss = loss.mean()
        return loss
