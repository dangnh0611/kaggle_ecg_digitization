from yagm.loss.mtl import MTLWeightedLoss
from torch import nn
import torch
from yagm.loss.heatmap_2d import (
    ChannelMaskedBiKLDivLoss2d,
    ChannelMaskedJSDLoss2d,
    ChannelMaskedBCEWithLogitsLoss2d,
)
from yagm.loss.regression import WeightedMSELoss, WeightedL1Loss, WeightedSmoothL1Loss


class KeypointDetectionLoss(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.loss

        # =========== main keypoints loss =========
        # heatmap loss
        heatmap_loss = cfg.main.heatmap_loss
        self._heatmap_mask = False
        if heatmap_loss == "cce":
            raise NotImplementedError
            # self.heatmap_loss = smp.losses.SoftCrossEntropyLoss(
            #     reduction='mean', smooth_factor=None, ignore_index=-100, dim=1)
        elif heatmap_loss == "bce":
            self.heatmap_main_loss_func = nn.BCEWithLogitsLoss(
                weight=None, reduction="mean", pos_weight=None
            )
        elif heatmap_loss == "mse":
            self.heatmap_main_loss_func = nn.MSELoss(reduction="mean")
        elif heatmap_loss == "l1":
            self.heatmap_main_loss_func = nn.L1Loss(reduction="mean")
        elif heatmap_loss == "kl":
            self._heatmap_mask = True
            self.heatmap_main_loss_func = ChannelMaskedBiKLDivLoss2d(
                reduction="masked_sum", weights=[1.0, 0.0]
            )
        elif heatmap_loss == "bikl":
            self._heatmap_mask = True
            self.heatmap_main_loss_func = ChannelMaskedBiKLDivLoss2d(
                reduction="masked_sum", weights=[0.5, 0.5]
            )
        elif heatmap_loss == "jsd":
            self._heatmap_mask = True
            self.heatmap_main_loss_func = ChannelMaskedJSDLoss2d(reduction="masked_sum")
        else:
            raise ValueError

        # Regression loss
        if cfg.main.reg_loss == "mse":
            self.reg_loss_func = WeightedMSELoss()
        elif cfg.main.reg_loss == "l1":
            self.reg_loss_func = WeightedL1Loss()
        elif cfg.main.reg_loss == "smooth_l1":
            self.reg_loss_func = WeightedSmoothL1Loss()
        else:
            raise ValueError

        # =========== grid heatmap loss =========
        if cfg.grid.heatmap_loss == "bce":
            self.heatmap_grid_loss_func = nn.BCEWithLogitsLoss(
                weight=None, reduction="mean", pos_weight=cfg.grid.pos_weight
            )
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
        all_kptness_gt,
        heatmap_all_pred,
        heatmap_all_gt,
        reg_main_kpt_pred,
        dsnt_main_kpt_pred,
        all_kpt_gt,
    ):
        loss_dict = {}
        grid_kptness_gt = all_kptness_gt[:, :-57]
        main_kptness_gt = all_kptness_gt[:, -57:]
        main_kpt_gt = all_kpt_gt[:, -57:]

        # main heatmap loss
        if not self._heatmap_mask:
            heatmap_main_loss = self.heatmap_main_loss_func(
                heatmap_all_pred[:, 1:], heatmap_all_gt[:, 1:]
            )
        else:
            heatmap_main_loss = self.heatmap_main_loss_func(
                heatmap_all_pred[:, 1:],
                heatmap_all_gt[:, 1:],
                main_kptness_gt.bool(),
            )
        loss_dict["heatmap_main_loss"] = heatmap_main_loss

        # main dsnt loss
        if self.loss_weights[1] > 0:
            assert dsnt_main_kpt_pred.shape[1] == 57
            dsnt_main_loss = self.reg_loss_func(
                dsnt_main_kpt_pred, main_kpt_gt, main_kptness_gt[..., None]
            )
            # if torch.isnan(dsnt_main_loss):
            #     print(
            #         torch.isnan(dsnt_main_kpt_pred).any(),
            #         torch.isnan(all_kpt_gt).any(),
            #         torch.isnan(all_kptness_gt).any(),
            #     )
            loss_dict["dsnt_main_loss"] = dsnt_main_loss
        else:
            dsnt_main_loss = None

        # main regression loss
        if self.loss_weights[2] > 0:
            assert reg_main_kpt_pred.shape[1] == 57
            reg_main_loss = self.reg_loss_func(
                reg_main_kpt_pred, main_kpt_gt, main_kptness_gt[..., None]
            )
            loss_dict["reg_main_loss"] = reg_main_loss
        else:
            reg_main_loss = None

        # grid heatmap loss
        if self.loss_weights[3] > 0:
            heatmap_grid_loss = self.heatmap_grid_loss_func(
                heatmap_all_pred[:, 0:1], heatmap_all_gt[:, 0:1]
            )
            loss_dict["heatmap_grid_loss"] = heatmap_grid_loss
        else:
            heatmap_grid_loss = None

        # total loss
        total_loss, _ = self.mtl(
            heatmap_main_loss,
            dsnt_main_loss,
            reg_main_loss,
            heatmap_grid_loss,
        )
        loss_dict["loss"] = total_loss

        return loss_dict
