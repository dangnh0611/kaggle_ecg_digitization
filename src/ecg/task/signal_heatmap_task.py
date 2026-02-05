import gc
import itertools
import logging
import os
import pickle
import time
from copy import deepcopy
from typing import Any, Dict, List

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tabulate import tabulate
from torch.nn import functional as F
from tqdm import tqdm
from yagm.data.datasets.base_dataset import BaseDataset
from yagm.layers.ops import group_max, group_mean, group_sum
from yagm.tasks.base_task import BaseTask
from yagm.transforms import monai_custom as CT
from yagm.transforms.keypoints.decode import decode_heatmap_2d_batched
from yagm.transforms.monai_custom import CustomProbNMS
from yagm.utils import misc as misc_utils
from yagm.utils.geometric import batch_perspective_transform_2d
from yagm.utils.misc import get_xlsx_copiable_metrics

from ecg.utils.comp_metrics_fast import compute_metrics
from ecg.utils.interpolate import interp_1d_cv2
from ecg.utils.viz import viz_img_heatmap, viz_img_keypoints, viz_pred_ecg_img_keypoints

logger = logging.getLogger(__name__)


class SignalHeatmapTask(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)

        loss_weights = cfg.loss.mtl.loss_weights
        assert len(loss_weights) == 3

        # placeholder for codec
        self.codec = None

    def build_loss_func(self):
        from ecg.loss.signal_heatmap_loss import SignalHeatmapLoss

        loss_func = SignalHeatmapLoss(self.cfg)
        logger.info("--------LOSS FUNCTION--------\n%s", loss_func)
        return loss_func

    def on_validation_start(self):
        logger.info("ON VALIDATION START")
        val_dataset = self.trainer.val_dataloaders.dataset
        if self.codec is None:
            self.codec = val_dataset.codec
        return super().on_validation_start()

    def on_validation_end(self):
        logger.info("ON VALIDATION END")
        # val_dataset = self.trainer.val_dataloaders.dataset
        return super().on_validation_end()

    def on_test_start(self):
        test_dataset = self.trainer.test_dataloaders.dataset
        if self.codec is None:
            self.codec = test_dataset.codec
        return super().on_test_start()

    def on_test_end(self):
        # test_dataset = self.trainer.test_dataloaders.dataset
        return super().on_test_end()

    def forward(self, image):
        return self.model(image)

    def shared_step(self, model, stage, batch, batch_idx, dataloader_idx=None):
        # print('INPUT BATCH:', {k: [v.shape, v.dtype] for k, v in batch.items()})

        batch_img = batch["image"]
        prob_heatmap_pred, offset_heatmap_pred, regress_signal_pred = model(batch_img)

        loss_dict = self.loss_func(
            prob_heatmap_pred,
            batch["prob_heatmap_gt"],
            offset_heatmap_pred,
            batch.get("offset_heatmap_gt", None),
            regress_signal_pred,
            batch["regress_signal_gt"],
        )

        step_output = {"loss": loss_dict["loss"]}
        step_metrics = {f"{stage}/{k}": v.item() for k, v in loss_dict.items()}
        if stage != "train":
            step_output["prob_heatmap_pred"] = self.model.heatmap_act(prob_heatmap_pred)
            step_output["offset_heatmap_pred"] = offset_heatmap_pred
            step_output["regress_signal_pred"] = regress_signal_pred

        return step_output, step_metrics

    def decode_prediction(self, batch, step_output):
        # decode the probability heatmap
        heatmap_signal_pred, regress_signal_pred = self.codec.batch_decode(
            step_output["prob_heatmap_pred"],
            step_output["offset_heatmap_pred"],
            step_output["regress_signal_pred"],
        )

        return {
            "heatmap_signal_pred": heatmap_signal_pred,
            "regress_signal_pred": regress_signal_pred,
        }

    def eval_step_single_model(
        self, model, batch, batch_idx, dataloader_idx=None, stage="val", model_name=None
    ):
        step_output, step_metrics = self.shared_step(
            model, stage, batch, batch_idx, dataloader_idx=dataloader_idx
        )

        step_save_output = self.decode_prediction(batch, step_output)
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
        def _nan_fill(arr, value):
            num_nan = np.isnan(arr).any(axis=-1).sum()
            if num_nan > 0:
                logger.warning("Detect %s nan predictions, fill to %f", num_nan, value)
                return np.nan_to_num(arr, nan=value)
            return arr

        if stage == "val":
            dataset = self.trainer.val_dataloaders.dataset
        elif stage == "test":
            dataset = self.trainer.test_dataloaders.dataset
        else:
            raise ValueError

        val_samples = dataset.oversamples
        val_gt_signals = dataset.cached_gt_signals

        all_gts = {}
        all_heatmap_preds = {}
        all_regress_preds = {}

        cur_idx = -1
        cur_sample_type_id = None
        for step_output in step_outputs:
            # .float() to deal with numpy >> << bf16
            heatmap_signal_pred = step_output["heatmap_signal_pred"].cpu().float().numpy()
            regress_signal_pred = step_output["regress_signal_pred"].cpu().float().numpy()
            assert heatmap_signal_pred.shape == regress_signal_pred.shape
            for i in range(regress_signal_pred.shape[0]):
                cur_idx += 1
                crop_name, sample_id, type_id = val_samples[cur_idx]
                sample_type_id = f"{sample_id}_{type_id}"
                sample_gt_signals = val_gt_signals[sample_id]
                len10 = len(sample_gt_signals["II"])

                if crop_name in ["II_short", "II_long_0", "II_long_2"]:
                    # 10250 // 4 = 2562
                    gt_length = len10 // 4
                elif crop_name in ["II_long_1", "II_long_3"]:
                    # 10250 // 4 + 1 = 2563
                    gt_length = len10 // 2 - len10 // 4
                else:
                    gt_length = len(sample_gt_signals[crop_name])
                    assert abs(gt_length - len10 // 4) <= 1

                # interpolate to original length
                ori_heatmap_signal_pred = interp_1d_cv2(
                    heatmap_signal_pred[i], gt_length, method="adaptive"
                )
                ori_regress_signal_pred = interp_1d_cv2(
                    regress_signal_pred[i], gt_length, method="adaptive"
                )

                # just reference, no additional RAM, don't worry
                all_gts[sample_type_id] = sample_gt_signals
                all_heatmap_preds.setdefault(sample_type_id, {})[
                    crop_name
                ] = ori_heatmap_signal_pred
                all_regress_preds.setdefault(sample_type_id, {})[
                    crop_name
                ] = ori_regress_signal_pred

        # post processing
        # merge multiple II_long crop into one
        # ensemble (e.g, average) II_short vs II_long
        all_fs = {}
        for sample_type_id, gt in all_gts.items():
            len10 = len(gt["II"])
            assert len10 % 10 == 0
            all_fs[sample_type_id] = len10 // 10

            heatmap_preds = all_heatmap_preds[sample_type_id]
            heatmap_preds["II"] = np.concatenate(
                [
                    heatmap_preds.pop(k)
                    for k in ("II_long_0", "II_long_1", "II_long_2", "II_long_3")
                ],
                axis=0,
            )

            regress_preds = all_regress_preds[sample_type_id]
            regress_preds["II"] = np.concatenate(
                [
                    regress_preds.pop(k)
                    for k in ("II_long_0", "II_long_1", "II_long_2", "II_long_3")
                ],
                axis=0,
            )
            assert len(heatmap_preds) == len(regress_preds) == 13

        # save predictions
        if self.cfg.task.save_pred:
            raise NotImplementedError
            save_dir = self.cfg.task.save_pred_dir or self.cfg.env.output_metadata_dir
            pred_save_path = os.path.join(
                save_dir, f"fold{self.cfg.cv.fold_idx}_predictions.pkl"
            )
            os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
            with open(pred_save_path, "wb") as f:
                pickle.dump(
                    {
                        "all_gts": all_gts,
                        "all_heatmap_preds": all_heatmap_preds,
                        "all_regress_preds": all_regress_preds,
                    },
                    f,
                )
            logger.info(
                "Saved fold %d predictions to %s", self.cfg.cv.fold_idx, pred_save_path
            )

        ###############################
        # EVALUATE
        metrics = {}
        best_method, best_snr = None, -999999
        per_sample_snr_df = pd.DataFrame({"sample_type_id": list(all_gts.keys())})
        per_sample_snr_df["sample_id"] = per_sample_snr_df["sample_type_id"].apply(
            lambda x: x.split("_")[0]
        )
        per_sample_snr_df["type_id"] = per_sample_snr_df["sample_type_id"].apply(
            lambda x: x.split("_")[1]
        )

        logger.info("Start computing metrics..")
        for method_name, method_preds in [
            ("heatmap", all_heatmap_preds),
            ("regress", all_regress_preds),
        ]:
            method_global_snr, method_per_sample_snrs = compute_metrics(
                method_preds,
                all_gts,
                all_fs,
                do_align=True,
            )
            metrics[f"{method_name}__SNR"] = method_global_snr
            if method_global_snr >= best_snr:
                best_method = method_name
                best_snr = method_global_snr
            per_sample_snr_df[method_name] = [
                method_per_sample_snrs[k] for k in all_gts.keys()
            ]
        metrics["SNR"] = best_snr
        logger.info("Done computing metrics!")

        # print(per_sample_snr_df)
        per_type_snr_df = per_sample_snr_df.groupby("type_id", as_index=False).agg(
            {"sample_id": "first", "heatmap": "mean", "regress": "mean"}
        )
        per_type_snr_df["heatmap"] = per_type_snr_df["heatmap"].apply(
            lambda x: max(10 * np.log10(x), -384.0)
        )
        per_type_snr_df["regress"] = per_type_snr_df["regress"].apply(
            lambda x: max(10 * np.log10(x), -384.0)
        )

        for _, row in per_type_snr_df.iterrows():
            type_id = row["type_id"]
            metrics[f"type_{type_id}_SNR"] = row[best_method]

        metrics = {f"{stage}/{k}": v for k, v in metrics.items()}

        # save everything to a dataframe
        metadata = {"per_sample_snr_df": per_sample_snr_df}
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
        super().on_test_epoch_end()

    def on_predict_epoch_end(self):
        raise NotImplementedError

    def get_single_fold_results(self):
        ret = {"cfg": self.cfg}
        all_best_metrics = self.metrics_tracker.all_best_metrics
        # mode, value, epoch, step, model, metadata, state
        best_metrics = all_best_metrics[self.cfg.task.metric][0]
        ret.update(best_metrics)
        return ret

    def oof_eval(self, all_fold_results, cache=None):
        raise NotImplementedError
