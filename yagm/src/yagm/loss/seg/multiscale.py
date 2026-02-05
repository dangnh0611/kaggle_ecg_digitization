from torch import nn
import logging
from torch.nn import functional as F
from yagm.loss.mtl import MTLWeightedLoss

logger = logging.getLogger(__name__)


class MultiscaleSegmentationLossWrapper(nn.Module):
    def __init__(
        self, loss_func, num_losses=1, mtl_cfg=None, interpolation="trilinear"
    ):
        super().__init__()
        self.loss_func = loss_func
        self.mtl = MTLWeightedLoss(
            weight_method=getattr(mtl_cfg, "method", "scalar"),
            loss_weights=getattr(mtl_cfg, "weights", [1.0] * num_losses),
            loss_names=getattr(mtl_cfg, "names", None),
            task_coefs=getattr(mtl_cfg, "task_coefs", [1.0] * num_losses),
            scale_invariant=getattr(mtl_cfg, "si", False),
        )
        self.interpolation = interpolation

    def get_coefs(self):
        if hasattr(self.mtl, "get_coefs"):
            return self.mtl.get_coefs()
        else:
            return None

    def set_epoch(self, epoch):
        if hasattr(self.mtl, "set_epoch"):
            self.mtl.set_epoch(epoch)

    def forward(self, ms_preds, target):
        ret = {}
        ms_losses = []
        for scale_idx, cur_scale_pred in enumerate(ms_preds):
            if cur_scale_pred.shape != target.shape:
                logger.debug(
                    "MS LOSS INTERPOLATE: %s --> %s", target.shape, cur_scale_pred.shape
                )
                cur_scale_target = F.interpolate(
                    target,
                    cur_scale_pred.shape[2:],
                    mode=self.interpolation,
                    align_corners=False,
                )
            else:
                cur_scale_target = target
                logger.debug("MS LOSS SAME: %s", target.shape)

            loss = self.loss_func(cur_scale_pred, cur_scale_target)
            if isinstance(loss, dict):
                # dictionary of different loss components, e.g BCE + Tversky combo
                cur_scale_loss = loss["loss"]
                for k, v in loss.items():
                    ret[f"{k}_{scale_idx}"] = v
            else:
                # tensor with 1 element
                cur_scale_loss = loss
                ret[f"loss_{scale_idx}"] = loss
            ms_losses.append(cur_scale_loss)
        loss, _ = self.mtl(*ms_losses)
        ret["loss"] = loss
        return ret
