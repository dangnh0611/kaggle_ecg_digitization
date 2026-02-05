import gc
import itertools
import logging
import os
import time
from copy import deepcopy
from typing import Any, Dict, List

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tabulate import tabulate
from torch.nn import functional as F
from tqdm import tqdm
from yagm.data.datasets.base_dataset import BaseDataset
from yagm.metrics.utils import plot_confusion_matrix
from yagm.tasks.base_task import BaseTask
from yagm.utils.misc import get_xlsx_copiable_metrics

from ecg.utils.image_rotation_metrics import compute_metrics

logger = logging.getLogger(__name__)


class ImageRotationTask(BaseTask):
    """
    ECG Image Rotation Classification
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_loss_func(self):
        from ecg.loss.image_rotation_loss import ImageRotationLoss

        return ImageRotationLoss(self.cfg)

    def on_validation_start(self):
        logger.info("ON VALIDATION START")
        # val_dataset = self.trainer.val_dataloaders.dataset
        return super().on_validation_start()

    def on_validation_end(self):
        logger.info("ON VALIDATION END")
        # val_dataset = self.trainer.val_dataloaders.dataset
        return super().on_validation_end()

    def on_test_start(self):
        # test_dataset = self.trainer.test_dataloaders.dataset
        return super().on_test_start()

    def on_test_end(self):
        # test_dataset = self.trainer.test_dataloaders.dataset
        return super().on_test_end()

    def forward(self, image):
        return self.model(image)

    def shared_step(self, model, stage, batch, batch_idx, dataloader_idx=None):
        # print('INPUT BATCH:', {k: [v.shape, v.dtype] for k, v in batch.items()})
        batch_img = batch["image"]
        gt_cls = batch["label"]
        gt_reg = batch["sine_cosine"]
        pred_cls_logit, pred_reg_logit = model(batch_img.float())
        loss = self.loss_func(pred_cls_logit, gt_cls, pred_reg_logit, gt_reg)
        # patch so that loss function always results in a dictionary
        if isinstance(loss, dict):
            loss_dict = loss
        elif not isinstance(loss, torch.Tensor):
            loss_dict = {"loss": loss}
        else:
            raise ValueError

        step_output = {"loss": loss_dict["loss"]}
        step_metrics = {f"{stage}/{k}": v.item() for k, v in loss_dict.items()}
        if stage != "train":
            step_output["pred_cls"] = F.softmax(pred_cls_logit, dim=-1)
            step_output["pred_reg"] = pred_reg_logit
        return step_output, step_metrics

    def eval_step_single_model(
        self, model, batch, batch_idx, dataloader_idx=None, stage="val", model_name=None
    ):

        step_output, step_metrics = self.shared_step(
            model, stage, batch, batch_idx, dataloader_idx=dataloader_idx
        )

        # save step output
        # need .float() or .half() to deal with BF16 Tensor to Numpy
        step_save_output = {
            "idx": batch["idx"].cpu().numpy(),
            "label": batch["label"].cpu().numpy(),
            "sine_cosine": batch["sine_cosine"].cpu().numpy(),
            "angle": batch["angle"].cpu().numpy(),
            "pred_cls": step_output["pred_cls"].float().cpu().numpy(),
            "pred_reg": step_output["pred_reg"].float().cpu().numpy(),
        }
        return step_output, step_save_output, step_metrics

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        return super().training_step(batch, batch_idx, dataloader_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return super().validation_step(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError

    def on_train_epoch_end(self):
        return super().on_train_epoch_end()

    def compute_metrics(
        self,
        step_outputs: List[Dict],
        dataset: BaseDataset,
        stage: str = "val",
        model_name=None,
    ):
        gt_cls = np.concatenate([e["label"] for e in step_outputs], axis=0)
        gt_sine_cosine = np.concatenate(
            [e["sine_cosine"] for e in step_outputs], axis=0
        )
        gt_angle = np.concatenate([e["angle"] for e in step_outputs], axis=0)
        pred_cls_prob = np.concatenate([e["pred_cls"] for e in step_outputs], axis=0)
        pred_reg = np.concatenate([e["pred_reg"] for e in step_outputs], axis=0)

        def _nan_fill(preds, value):
            num_nan = np.isnan(preds).any(axis=-1).sum()
            if num_nan > 0:
                logger.warning("Detect %s nan predictions, fill to %f", num_nan, value)
                return np.nan_to_num(preds, nan=value)
            return preds

        pred_cls_prob = _nan_fill(pred_cls_prob, 0.25)
        pred_reg = _nan_fill(pred_reg, 0.0)

        ###############################
        # EVALUATE
        metrics = compute_metrics(
            gt_cls, gt_angle, gt_sine_cosine, pred_cls_prob, pred_reg
        )
        cm = metrics.pop("confusion_matrix")
        cm_img = plot_confusion_matrix(
            cm, class_names=["0", "90", "180", "270"], number_fmt="d"
        )
        # save confusion matrix as an image
        cm_save_path = os.path.join(
            self.cfg.env.output_metadata_dir,
            "confusion_matrix",
            model_name,
            f"ep={self.current_epoch}_step={self.global_step}.jpg",
        )
        os.makedirs(os.path.dirname(cm_save_path), exist_ok=True)
        cv2.imwrite(cm_save_path, cv2.cvtColor(cm_img, cv2.COLOR_RGB2BGR))

        # log confusion matrix to experiment tracker, e.g Wandb
        for logger in self.loggers:
            if hasattr(logger, "log_image"):
                logger.log_image(
                    key=f"{stage}/confusion_matrix/{model_name}",
                    images=[cm_img],
                    caption=["Confusion Matrix"],
                )

        metrics = {f"{stage}/{k}": v for k, v in metrics.items()}

        # save everything to a dataframe
        pred_df = pd.DataFrame()
        pred_df["gt_cls"] = gt_cls
        pred_df["gt_angle"] = gt_angle
        pred_df[["gt_sine", "gt_cosine"]] = gt_sine_cosine
        pred_df[[f"pred_prob_{i}" for i in range(4)]] = pred_cls_prob
        pred_df["pred_cls"] = np.argmax(pred_cls_prob, axis=-1)
        pred_df[["pred_sine", "pred_cosine"]] = pred_reg
        metadata = {"pred_df": pred_df}
        return metrics, metadata

    def on_epoch_end_save_metadata(self, metadata, stage, model_name):
        super().on_epoch_end_save_metadata(metadata, stage, model_name)
        save_metadata = {}

        for k, v in metadata.items():
            save_path = os.path.join(
                self.cfg.env.output_metadata_dir,
                k,
                model_name,
                f"ep={self.current_epoch}_step={self.global_step}.csv",
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            v.to_csv(save_path, index=False)
            save_metadata[k] = save_path

        # add `stage`/ prefix, e.g val/something
        save_metadata = {f"{stage}/{k}": v for k, v in save_metadata.items()}
        return save_metadata

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        copiable_metrics_str = get_xlsx_copiable_metrics(
            self.metrics_tracker.all_best_metrics, self.cfg.task.log_metrics
        )
        logger.info(
            "COPIABLE METRICS at epoch=%f step=%d:\n%s",
            self.current_exact_epoch,
            self.global_step,
            copiable_metrics_str,
        )

    def on_test_epoch_end(self) -> None:
        # post processing
        # calculate & log metric
        super().on_test_epoch_end()

    def on_predict_epoch_end(self):
        print("On predict epoch end..")
        raise NotImplementedError

    def get_single_fold_results(self):
        ret = {}
        ret["metrics_trackers"] = self.metrics_tracker
        # grab all saved checkpoints
        best_ckpt_paths = []
        for cb in self.trainer.checkpoint_callbacks:
            best_ckpt_paths.extend(list(cb.best_k_models.keys()))
        best_ckpt_paths = list(set(best_ckpt_paths))

        # list of best checkpoints
        # each checkpoint is identified by its metadata dict
        # with keys: epoch, step
        best_ckpts = []
        if best_ckpt_paths:
            for ckpt_path in best_ckpt_paths:
                ckpt_name = os.path.basename(ckpt_path)
                epoch = int(ckpt_name.split("_")[0].replace("ep=", ""))
                step = int(ckpt_name.split("_")[1].replace("step=", ""))
                best_ckpts.append(
                    {"epoch": epoch, "step": step, "ckpt_path": ckpt_path}
                )
        else:
            # in case of validation/test
            assert not self.cfg.train
            best_ckpts.append({"epoch": 0, "step": 0})
        ret["best_ckpts"] = best_ckpts
        return ret
