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

from ecg.utils.metrics import compute_metrics
from ecg.utils.viz import viz_img_heatmap, viz_img_keypoints, viz_pred_ecg_img_keypoints

logger = logging.getLogger(__name__)


class KeypointDetectionTask(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)

        loss_weights = cfg.loss.mtl.loss_weights
        assert len(loss_weights) == 4
        # assert loss_weights[0] > 0
        self._enable_dsnt_head = loss_weights[1] > 0
        self._enable_sa_reg_head = loss_weights[2] > 0
        self._enable_heatmap_grid = loss_weights[3] > 0

        self.input_wh = np.array([cfg.data.img_size[1], cfg.data.img_size[0]])
        self.heatmap_stride_wh = np.array(
            [cfg.data.heatmap_stride[1], cfg.data.heatmap_stride[0]]
        )
        self.heatmap_wh = self.input_wh / self.heatmap_stride_wh
        logger.info(
            "input_wh=%s heatmap_stride_wh=%s heatmap_wh=%s",
            self.input_wh,
            self.heatmap_stride_wh,
            self.heatmap_wh,
        )
        # radius: ~40 on 1700x2200 image
        # assume ROI has 0.5x scale, image is isotropically resize + padding to `cfg.data.img_size`
        self.nms_radius_thres = (
            39.37007874015748
            * 0.5
            / max(1700 / self.heatmap_wh[1], 2200 / self.heatmap_wh[0])
        )
        logger.info("Using NMS radius threshold of %f", self.nms_radius_thres)

    def build_loss_func(self):
        from ecg.loss.keypoint_detection_loss import KeypointDetectionLoss

        loss_func = KeypointDetectionLoss(self.cfg)
        logger.info("--------LOSS FUNCTION--------\n%s", loss_func)
        return loss_func

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
        heatmap_all_pred, dsnt_main_kpt_pred, reg_main_kpt_pred = model(batch_img)

        loss_dict = self.loss_func(
            batch["kptness_gt"],
            heatmap_all_pred,
            batch["heatmap_gt"],
            reg_main_kpt_pred,
            dsnt_main_kpt_pred,
            batch["kpt_gt"],
        )

        step_output = {"loss": loss_dict["loss"]}
        step_metrics = {f"{stage}/{k}": v.item() for k, v in loss_dict.items()}
        if stage != "train":
            # for debug only
            # step_output["heatmap_all_pred"] = batch["heatmap_gt"]
            heatmap_all_pred[:, 0] = F.sigmoid(heatmap_all_pred[:, 0])
            step_output["heatmap_all_pred"] = heatmap_all_pred
            step_output["reg_main_kpt_pred"] = reg_main_kpt_pred
            step_output["dsnt_main_kpt_pred"] = dsnt_main_kpt_pred

        return step_output, step_metrics

    def decode_prediction(self, input_batch, step_output):
        ori_wh = input_batch["ori_wh"].cpu().numpy()  # (N, 2)
        batch_M = input_batch["M"].cpu().numpy()  # (N, 3, 3)
        decoded = {}

        # =================== HEATMAP DECODE =================
        # decode -> main points
        main_kpt_preds = {}
        # YX order in input image space
        batch_heatmap_main_kpt = decode_heatmap_2d_batched(
            step_output["heatmap_all_pred"][:, 1:],
            pool_ksize=[3, 3],
            nms_radius_thres=self.nms_radius_thres,
            blur_operator=None,
            conf_thres=None,  # just take argmax, don't care about confidence score
            max_dets=1,
            timeout=None,
        )
        # should has same shape (L, 3) for each element in the batch
        # we can pack it into single numpy array of shape (N, L, 1, 3)
        batch_heatmap_main_kpt = np.array(batch_heatmap_main_kpt)
        assert batch_heatmap_main_kpt.shape[2] == 1
        batch_heatmap_main_kpt = batch_heatmap_main_kpt[:, :, 0, :]  # (N, L, 3)
        assert batch_heatmap_main_kpt.shape[1:] == (57, 3)  # just 57 main keypoints
        # yx -> xy -> contigous coordinate
        xy = batch_heatmap_main_kpt[..., [1, 0]] + 0.5
        # back to input image space: (N, L, 2) * (1, 1, 2)
        xy = xy * self.heatmap_stride_wh[None, None]
        # back to original image space, shape (N, L, 2)
        xy = batch_perspective_transform_2d(
            xy,  # (N, L, 2)
            batch_M,  # (N, 3, 3)
        )
        # (x, y, conf) format
        heatmap_main_kpt_pred = np.concatenate(
            [xy, batch_heatmap_main_kpt[..., 2:3]], axis=2
        )  # (N, L, 3)
        main_kpt_preds["heatmap_main_kpt_pred"] = heatmap_main_kpt_pred

        if self._enable_heatmap_grid and self.cfg.task.decode.grid.enable:
            # decode -> grid points
            # YX order, heatmap pixel indices space
            _t0 = time.time()
            raw_heatmap_grid_kpt_pred = decode_heatmap_2d_batched(
                step_output["heatmap_all_pred"][:, 0:1],
                pool_ksize=[3, 3],
                nms_radius_thres=self.nms_radius_thres,
                blur_operator=None,
                conf_thres=0.05,
                max_dets=10_000,
                timeout=10,
            )
            _t1 = time.time()
            logger.debug("Decode grid points take %.2f sec", _t1 - _t0)
            # un-fixed number of grid keypoints detected for each image
            # so we need a for loop over each image/element of the batch
            heatmap_grid_kpt_pred = []
            for i, img_grid_kpt in enumerate(raw_heatmap_grid_kpt_pred):
                assert len(img_grid_kpt) == 1
                img_grid_kpt = img_grid_kpt[0].numpy()
                if len(img_grid_kpt) == 0:
                    # we met empty array of shape (0, 3) here
                    decoded_img_grid_kpt = img_grid_kpt
                else:
                    # (L, 3)
                    assert img_grid_kpt.shape[1] == 3
                    # yx -> xy -> contigous coordinate
                    xy = img_grid_kpt[:, [1, 0]] + 0.5
                    # back to input image space
                    xy = xy * self.heatmap_stride_wh[None]
                    # back to original image space
                    xy = batch_perspective_transform_2d(
                        xy[None],  # (1, L, 2)
                        batch_M[i : i + 1],  # (1, 3, 3)
                    )[0]
                    # (x, y, conf) format
                    decoded_img_grid_kpt = np.concatenate(
                        [xy, img_grid_kpt[:, 2:3]], axis=1
                    )
                heatmap_grid_kpt_pred.append(decoded_img_grid_kpt)
        else:
            # disable actual decoding, just return a placeholder array of numel=0
            heatmap_grid_kpt_pred = [
                np.zeros((0, 3), dtype=np.float32) for _ in range(batch_M.shape[0])
            ]
        decoded["heatmap_grid_kpt_pred"] = heatmap_grid_kpt_pred

        # ================= DECODE THE DSNT ================
        if self._enable_dsnt_head:
            dsnt_main_kpt_pred = step_output["dsnt_main_kpt_pred"].cpu().numpy()
            assert dsnt_main_kpt_pred.shape[1:] == (57, 2)
            # already contigous in range [0,1], xy order: (N, L, 2) * (1, 1, 2)
            dsnt_main_kpt_pred = dsnt_main_kpt_pred * self.input_wh[None, None]
            dsnt_main_kpt_pred = batch_perspective_transform_2d(
                dsnt_main_kpt_pred,  # (N, L, 2)
                batch_M,  # (N, 3, 3)
            )
            main_kpt_preds["dsnt_main_kpt_pred"] = dsnt_main_kpt_pred

        # ================= DECODE THE DIRECT REGRESSION =========
        if self._enable_sa_reg_head:
            reg_main_kpt_pred = step_output["reg_main_kpt_pred"].cpu().numpy()
            assert reg_main_kpt_pred.shape[1:] == (57, 2)
            # already contigous in range [0,1], xy order: (N, L, 2) * (1, 1, 2)
            reg_main_kpt_pred = reg_main_kpt_pred * self.input_wh[None, None]
            reg_main_kpt_pred = batch_perspective_transform_2d(
                reg_main_kpt_pred,  # (N, L, 2)
                batch_M,  # (N, 3, 3)
            )
            main_kpt_preds["reg_main_kpt_pred"] = reg_main_kpt_pred

        decoded.update(main_kpt_preds)

        # visualize if needed
        if self.cfg.task.viz.enable:
            dataset = self.trainer.val_dataloaders.dataset
            sample_idxs = input_batch["idx"].tolist()
            for i, sample_idx in enumerate(sample_idxs):
                sample = dataset.gt_df[sample_idx].to_dicts()[0]
                sample_id = sample["id"]
                type_id = sample["type_id"]
                rot_code = sample["rot_code"]
                # read the image
                img_name = f"{sample_id}-{type_id:04d}.png"
                img_path = os.path.join(dataset.img_dir, str(sample_id), img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # correct the orientation
                if rot_code is not None:
                    assert rot_code == int(rot_code)
                    rot_code = int(rot_code)
                    img = cv2.rotate(img, rot_code)

                # viz_kpt_xys = np.concatenate(
                #     [heatmap_grid_kpt_pred[i], heatmap_main_kpt_pred[i]], axis=0
                # )[:, :2]
                # viz_kpt_classes = ([0] * len(heatmap_grid_kpt_pred[i])) + list(
                #     range(1, 58)
                # )

                # viz_kpt_xys = input_batch['ori_kpt_gt'][i].cpu().numpy()
                # viz_kpt_classes = ([0] * 2365)  + list(range(1, 58))

                # viz_img_keypoints(
                #     img,
                #     kpt_xys=viz_kpt_xys,
                #     kpt_classes=viz_kpt_classes,
                #     save_path=f"outputs/tmp2/{img_name}",
                #     figsize=(16, 16),
                #     point_size=20,
                #     alpha=0.9,
                #     show_legend=False,
                # )

                if self.cfg.task.viz.draw_keypoints:
                    viz_pred_ecg_img_keypoints(
                        img,
                        grid_points=heatmap_grid_kpt_pred[i][..., :2],
                        main_points_dict={
                            k: v[i, :, :2] for k, v in main_kpt_preds.items()
                        },
                        save_path=f"outputs/viz/keypoint/{img_name}",
                        figsize=(32, 32),
                        point_size=30,
                        alpha=0.9,
                        show_legend=False,
                    )
                if self.cfg.task.viz.draw_heatmap:
                    heatmap = step_output["heatmap_all_pred"][i].clone()  # (C, H2, W2)
                    C, H2, W2 = heatmap.shape
                    # first channel already applied sigmoid()
                    # now apply spatial softmax to the rest channels
                    heatmap[1:] = F.softmax(
                        heatmap[1:].reshape(C - 1, -1), dim=1
                    ).reshape(C - 1, H2, W2)
                    viz_img_heatmap(
                        img.transpose(2, 0, 1),
                        heatmap.cpu().numpy(),
                        save_path=f"outputs/viz/heatmap/{img_name}",
                        figsize=(32, 32),
                        only_overlay=False,
                    )

        return decoded

    def eval_step_single_model(
        self, model, batch, batch_idx, dataloader_idx=None, stage="val", model_name=None
    ):
        step_output, step_metrics = self.shared_step(
            model, stage, batch, batch_idx, dataloader_idx=dataloader_idx
        )
        step_save_output = self.decode_prediction(batch, step_output)
        step_save_output.update(
            {
                "ori_kpt_gt": batch["ori_kpt_gt"].cpu().numpy(),
                "kptness_gt": batch["kptness_gt"].cpu().numpy(),
                "ori_scale_from_ref": batch["ori_scale_from_ref"].cpu().numpy(),
            }
        )
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
        def _nan_fill(preds, value):
            num_nan = np.isnan(preds).any(axis=-1).sum()
            if num_nan > 0:
                logger.warning("Detect %s nan predictions, fill to %f", num_nan, value)
                return np.nan_to_num(preds, nan=value)
            return preds

        ori_scale_from_ref = np.concatenate(
            [e["ori_scale_from_ref"] for e in step_outputs], axis=0
        )
        # threshold in term of Euclidean distance
        # defined in reference image (fixed shape/resolution)
        per_img_threshold = ori_scale_from_ref * self.cfg.task.eval.thres

        ori_kpt_gt = np.concatenate([e["ori_kpt_gt"] for e in step_outputs], axis=0)
        kptness_gt = np.concatenate([e["kptness_gt"] for e in step_outputs], axis=0)

        N = len(ori_kpt_gt)
        assert N == len(kptness_gt)

        # main keypoints dict
        main_kpt_preds = {}

        # main keypoints from heatmap
        heatmap_main_kpt_pred = np.concatenate(
            [e["heatmap_main_kpt_pred"] for e in step_outputs], axis=0
        )
        assert len(heatmap_main_kpt_pred) == N
        main_kpt_preds["heatmap"] = heatmap_main_kpt_pred

        # main keypoints from SA regression
        if self._enable_sa_reg_head:
            reg_main_kpt_pred = np.concatenate(
                [e["reg_main_kpt_pred"] for e in step_outputs], axis=0
            )
            assert len(reg_main_kpt_pred) == N
            main_kpt_preds["reg"] = reg_main_kpt_pred

        # main keypoints from DSNT
        if self._enable_dsnt_head:
            dsnt_main_kpt_pred = np.concatenate(
                [e["dsnt_main_kpt_pred"] for e in step_outputs], axis=0
            )
            assert len(dsnt_main_kpt_pred) == N
            main_kpt_preds["dsnt"] = dsnt_main_kpt_pred

        # heatmap points
        heatmap_grid_kpt_pred = []
        for step_output in step_outputs:
            heatmap_grid_kpt_pred.extend(step_output["heatmap_grid_kpt_pred"])
        assert len(heatmap_grid_kpt_pred) == N

        # save predictions
        if self.cfg.task.save_pred:
            save_dir = self.cfg.task.save_pred_dir or self.cfg.env.output_metadata_dir
            pred_save_path = os.path.join(
                save_dir, f"fold{self.cfg.cv.fold_idx}_predictions.pkl"
            )
            os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
            val_dataset = self.trainer.val_dataloaders.dataset
            with open(pred_save_path, "wb") as f:
                pickle.dump(
                    {
                        "gt_df": val_dataset.gt_df,
                        "main_kpt_preds": main_kpt_preds,
                        "grid_kpt_pred": heatmap_grid_kpt_pred,
                    },
                    f,
                )
            logger.info(
                "Saved fold %d predictions to %s", self.cfg.cv.fold_idx, pred_save_path
            )

        ###############################
        # EVALUATE
        metrics = compute_metrics(
            main_kpt_preds=main_kpt_preds,
            grid_kpt_pred=heatmap_grid_kpt_pred,
            kpt_gt=ori_kpt_gt,
            kptness_gt=kptness_gt,
            per_img_threshold=per_img_threshold,
        )

        for best_metric_key in ["AVG_ACC", "ACC"]:
            metrics[f"best__{best_metric_key}"] = max(
                [
                    metrics[f"{method}__{best_metric_key}"]
                    for method in main_kpt_preds.keys()
                ]
            )
        metrics = {f"{stage}/{k}": v for k, v in metrics.items()}

        # save everything to a dataframe
        metadata = {"pred_df": None}
        return metrics, metadata

    def on_epoch_end_save_metadata(self, metadata, stage, model_name):
        super().on_epoch_end_save_metadata(metadata, stage, model_name)
        save_metadata = {}

        for k, v in metadata.items():
            save_path = os.path.join(
                self.cfg.env.output_metadata_dir,
                k,
                model_name,
                f"ep={self.current_epoch}_step={self.global_step}.npy",
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # v.to_csv(save_path, index=False)
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
        logger.info("START OOF EVALUATION ON %d FOLDS", len(all_fold_results))
        print(all_fold_results)
        oof_pred_df = pd.DataFrame()
        for i, (fold_idx, fold_result) in enumerate(all_fold_results.items()):
            fold_cfg = fold_result["cfg"]
            assert fold_idx == int(fold_cfg.cv.fold_idx)
            fold_epoch = fold_result["epoch"]
            fold_step = fold_result["step"]
            pred_csv_path = fold_result["metadata"]["val/pred_df"]
            fold_pred_df = pd.read_csv(pred_csv_path)
            assert "fold" not in fold_pred_df.columns
            fold_pred_df["fold"] = fold_idx
            oof_pred_df = pd.concat([oof_pred_df, fold_pred_df])
            logger.info(
                "%d/%d fold=%d epoch=%d step=%d, load prediction dataframe of shape %s (OOF TOTAL %s) from %s",
                i + 1,
                len(all_fold_results),
                fold_idx,
                fold_epoch,
                fold_step,
                fold_pred_df.shape,
                oof_pred_df.shape,
                pred_csv_path,
            )

        # compute OOF metrics
        gt_target5 = oof_pred_df[list(constants.TARGET_NAMES)].to_numpy()

        oof_metrics = {}
        best_metrics = {}
        eval_methods = []
        # 9 losses: ['sa_target5', 'composed_target3', 'target5', 'area_scale', 'date', 'ndni', 'height', 'state', 'species']
        loss_weights = self.cfg.loss.mtl.loss_weights
        if loss_weights[0] > 0:
            eval_methods.append("sa")
        if loss_weights[1] > 0:
            eval_methods.append("composed")
        if loss_weights[2] > 0:
            eval_methods.append("pooled")

        for method in eval_methods:
            assert method != "best"
            method_pred5 = oof_pred_df[
                [f"{method}__{target_name}" for target_name in constants.TARGET_NAMES]
            ].to_numpy()
            method_metrics = compute_metrics(method_pred5, gt_target5)
            for k, v in method_metrics.items():
                oof_metrics[f"{method}__{k}"] = v
                best_metrics[k] = max(best_metrics.get(k, -9999), v)
        # update best ones
        oof_metrics.update({f"best__{k}": v for k, v in best_metrics.items()})
        oof_metrics = {f"val/{k}": v for k, v in oof_metrics.items()}
        logger.info("OOF METRICS:\n%s", oof_metrics)
        return [oof_metrics]
