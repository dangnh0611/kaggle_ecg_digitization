from torch import nn
from torch.nn import functional as F
import torch


class HardExampleMiningLossWrapper(nn.Module):
    """
    Wrap a pixelwise/pointwise loss function with FocalLoss-like weighting.
    Support two variants:
        - focal: original Focal loss mentioned in original paper
            + paper: https://arxiv.org/abs/1708.02002v2
            + ref: https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        - wahr: Weight-Adaptive Heatmap Regression
            + paper: https://arxiv.org/abs/2012.15175
            + ref: https://github.com/greatlog/SWAHR-HumanPose/blob/master/lib/core/loss.py#L29
        - wahr_v2: patched version with solve some posible issues of WAHR
    """

    def __init__(
        self, loss_func, alpha=None, gamma=1.5, method=None, act="sigmoid", detach=False
    ):
        """
        Args:
            loss_func: A pixelwise/pointwise loss module, i.e nn.BCEWithLogitsLoss(reduction=None)
            alpha: target-based reweighting coef, higher -> prioritize foreground, lower -> prioritize background
            gamma: exponential coef gamma used in Focal loss
            method: one of focal, wahr, wahr_v2
            act: activation must be applied to convert logits to probabilities
            detach: should apply `.detach()` to the weighting term, so HEM weighting term won't
                contribute to the gradient. Default to False as the official implementation.
                More details:
                    - https://github.com/kuangliu/pytorch-retinanet/issues/12
                    - https://discuss.pytorch.org/t/how-to-implement-focal-loss-in-pytorch/6469
        """
        super().__init__()
        assert act in ["sigmoid", "identity"]
        assert method in ["focal", "wahr", "wahr_v2", None]
        self.loss_func = loss_func
        self.alpha = alpha
        self.gamma = gamma
        self.method = method
        self.act = act
        self.detach = detach

    def forward(self, pred, target):
        alpha, gamma = self.alpha, self.gamma
        pw_loss = self.loss_func(pred, target)

        if self.method is not None:
            if self.act == "sigmoid":
                pred = pred.sigmoid()  # prob from logits

            if self.method == "focal":
                p_t = target * torch.abs(pred) + (1 - target) * torch.abs(1 - pred)
                hem_weight = (1.0 - p_t) ** gamma
            elif self.method == "wahr":
                # Problematic with soft-label such as gaussian heatmap
                # Event if accurate prediction (very near groundtruth), still assigned high weight
                # which, prioritize or give higher weight to pixels with low pred/gt value
                h_gamma = target**gamma
                # equivalent to `h_gamma + pred * (1 - 2 * h_gamma)`
                hem_weight = (
                    torch.abs(pred) * (1 - h_gamma) + torch.abs(1 - pred) * h_gamma
                )
            elif self.method == "wahr_v2":
                # this solve the above issue of WAHR
                h_gamma = target**gamma
                # equivalent to pred + target * h_gamma - 2 * pred * h_gamma
                # pred - 2 * pred * h_gamma = pred * (1 - 2 * h_gamma)
                hem_weight = (
                    torch.abs(pred) * (1 - h_gamma) + torch.abs(target - pred) * h_gamma
                )
            else:
                raise ValueError
            if hem_weight is not None:
                if self.detach:
                    hem_weight = hem_weight.detach()
                pw_loss *= hem_weight

        if alpha is not None:
            alpha_factor = target * alpha + (1 - target) * (1 - alpha)
            pw_loss *= alpha_factor

        return pw_loss.mean()
