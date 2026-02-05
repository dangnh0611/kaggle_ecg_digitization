import torch
from torch import nn
from torch.nn import functional as F
from yagm.loss.heatmap_2d import (
    ChannelMaskedBCEWithLogitsLoss2d,
    ChannelMaskedBiKLDivLoss2d,
    ChannelMaskedJSDLoss2d,
)
from yagm.loss.mtl import MTLWeightedLoss
from yagm.loss.regression import WeightedL1Loss, WeightedMSELoss, WeightedSmoothL1Loss
from yagm.loss.seg.soft_cce import SoftCrossEntropyLoss


class MaskedColumnCCELoss(nn.Module):

    def __init__(
        self,
        weight=None,
        ignore_index=-100,
        reduction="masked_sum",
        label_smoothing=0.0,
    ):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input, target, mask=None):
        N, C, H, W = target.shape

        # normalize target to a probability distribution
        target = target / target.sum(dim=2, keepdim=True)

        if mask is not None:
            input = torch.masked_fill(input, ~mask[..., None, None], 1.0)
            target = torch.masked_fill(target, ~mask[..., None, None], 1.0)

        if self.reduction == "masked_sum":
            _reduction = "sum"
            if mask is not None:
                mask_sum = mask.sum()
            else:
                mask_sum = N * C * W
        else:
            _reduction = self.reduction

        # NCHW -> NHCW
        input = input.permute(0, 2, 1, 3)
        target = target.permute(0, 2, 1, 3)
        loss = F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=_reduction,
            label_smoothing=self.label_smoothing,
        )
        if self.reduction == "masked_sum":
            loss = loss / mask_sum
        return loss


class SignalHeatmapLoss(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.loss
        self.GT_W, self.GT_L = global_cfg.data.heatmap_hwl[1:]
        self.LPAD = (self.GT_W - self.GT_L) // 2
        self.RPAD = self.GT_W - self.GT_L - self.LPAD
        assert self.RPAD == self.LPAD

        # =========== main keypoints loss =========
        # heatmap loss
        heatmap_loss = cfg.heatmap_loss
        self._heatmap_loss = heatmap_loss
        self._heatmap_mask = False
        if heatmap_loss == "cce":
            self.heatmap_loss_func = MaskedColumnCCELoss(
                reduction="masked_sum", label_smoothing=0.0
            )
        elif heatmap_loss == "bce":
            self.heatmap_loss_func = nn.BCEWithLogitsLoss(
                weight=None, reduction="mean", pos_weight=None
            )
        elif heatmap_loss == "mse":
            self.heatmap_loss_func = nn.MSELoss(reduction="mean")
        elif heatmap_loss == "l1":
            self.heatmap_loss_func = nn.L1Loss(reduction="mean")
        elif heatmap_loss == "kl":
            self._heatmap_mask = True
            self.heatmap_loss_func = ChannelMaskedBiKLDivLoss2d(
                reduction="masked_sum", dim=(2,), weights=[1.0, 0.0]
            )
        elif heatmap_loss == "bikl":
            self._heatmap_mask = True
            self.heatmap_loss_func = ChannelMaskedBiKLDivLoss2d(
                reduction="masked_sum", dim=(2,), weights=[0.5, 0.5]
            )
        elif heatmap_loss == "jsd":
            self._heatmap_mask = True
            self.heatmap_loss_func = ChannelMaskedJSDLoss2d(
                reduction="masked_sum", dim=(2,)
            )
        else:
            raise ValueError

        # Regression loss
        if cfg.regress_loss == "mse":
            self.regress_loss_func = WeightedMSELoss()
        elif cfg.regress_loss == "l1":
            self.regress_loss_func = WeightedL1Loss()
        elif cfg.regress_loss == "smooth_l1":
            self.regress_loss_func = WeightedSmoothL1Loss()
        else:
            raise ValueError

        # ============ MTL ===========
        self.mtl = MTLWeightedLoss(**cfg.mtl)
        loss_weights = self.mtl.loss_weights
        self.loss_weights = loss_weights

    def set_epoch(self, epoch):
        if hasattr(self.mtl, "set_epoch"):
            self.mtl.set_epoch(epoch)

    def forward(
        self,
        prob_heatmap_pred,
        prob_heatmap_gt,
        offset_heatmap_pred,
        offset_heatmap_gt,
        regress_signal_pred,
        regress_signal_gt,
    ):
        loss_dict = {}

        # heatmap prob loss
        if not self._heatmap_mask:
            heatmap_loss = self.heatmap_loss_func(
                prob_heatmap_pred[:, :, :, self.LPAD : -self.RPAD],
                prob_heatmap_gt[:, :, :, self.LPAD : -self.RPAD],
            )
        else:
            heatmap_loss = self.heatmap_loss_func(
                prob_heatmap_pred[:, :, :, self.LPAD : -self.RPAD],
                prob_heatmap_gt[:, :, :, self.LPAD : -self.RPAD],
                None,
            )
        loss_dict["heatmap_loss"] = heatmap_loss

        # print('PRED:', torch.amax(F.softmax(prob_heatmap_pred[:, :, :, self.LPAD : -self.RPAD], dim=2), dim=2)[0, 0])
        # print('GT:', torch.amax(prob_heatmap_gt[:, :, :, self.LPAD : -self.RPAD], dim=2)[0, 0])
        # print('GT SUM:', torch.sum(prob_heatmap_gt[:, :, :, self.LPAD : -self.RPAD], dim=2)[0, 0])

        # heatmap offset loss
        if self.loss_weights[1] > 0:
            raise NotImplementedError
            offset_loss = self.offset_loss_func()
        else:
            offset_loss = None

        # main dsnt loss
        if self.loss_weights[2] > 0:
            assert (
                regress_signal_pred.shape[1] == 1
                and regress_signal_pred.shape[2] == self.GT_W
            )
            assert (
                regress_signal_gt.ndim == 2 and regress_signal_gt.shape[1] == self.GT_L
            )
            # (N, GT_W) vs (N, GT_L)
            regress_loss = self.regress_loss_func(
                regress_signal_pred[:, 0, self.LPAD : -self.RPAD],
                regress_signal_gt.to(regress_signal_pred),
                None,
            )
            loss_dict["regress_loss"] = regress_loss
        else:
            regress_loss = None

        # total loss
        total_loss, _ = self.mtl(
            heatmap_loss,
            offset_loss,
            regress_loss,
        )
        loss_dict["loss"] = total_loss

        return loss_dict
