import logging
from copy import deepcopy

import hydra
from torch import nn
from torch.nn import functional as F
from yagm.loss.mtl import MTLWeightedLoss

logger = logging.getLogger(__name__)


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, reduction="mean", pos_weight=None):
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32).reshape(
                1, -1, 1, 1, 1
            )  # BCHWD
        super().__init__(reduction=reduction, pos_weight=pos_weight)


class ImageRotationLoss(nn.Module):
    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.loss

        # CLASSIFICATION LOSS
        self.cls_loss_func = hydra.utils.instantiate(cfg.cls)
        # REGRESSION LOSS
        self.reg_loss_func = hydra.utils.instantiate(cfg.reg)

        # MULTI-TASK WEIGHTING SCHEME
        mtl_cfg = cfg.mtl
        self.mtl = MTLWeightedLoss(
            weight_method=getattr(mtl_cfg, "method", "scalar"),
            loss_weights=getattr(mtl_cfg, "weights", [1.0] * 2),
            loss_names=getattr(mtl_cfg, "names", None),
            task_coefs=getattr(mtl_cfg, "task_coefs", [1.0] * 2),
            scale_invariant=getattr(mtl_cfg, "si", False),
        )

    def get_coefs(self):
        if hasattr(self.mtl, "get_coefs"):
            return self.mtl.get_coefs()
        else:
            return None

    def set_epoch(self, epoch):
        if hasattr(self.mtl, "set_epoch"):
            self.mtl.set_epoch(epoch)

    def forward(self, cls_pred, cls_target, reg_pred, reg_target):
        ret = {}
        # classification
        cls_loss = self.cls_loss_func(cls_pred, cls_target)
        ret["cls_loss"] = cls_loss

        # regression
        reg_loss = self.reg_loss_func(reg_pred, reg_target)
        ret["reg_loss"] = reg_loss

        # combine
        loss, _ = self.mtl(cls_loss, reg_loss)
        ret["loss"] = loss
        return ret
