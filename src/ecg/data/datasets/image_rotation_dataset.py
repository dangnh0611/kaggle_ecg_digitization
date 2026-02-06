import gc
import json
import logging
import math
import os
import random
import time
from functools import partial
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import hydra
import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf
from scipy.stats import truncnorm
from torch.nn import functional as F
from yagm.data.datasets.base_dataset import BaseDataset
from yagm.transforms import albumentations_custom as AC
from yagm.transforms.keypoints.encode import (
    generate_2d_gaussian_heatmap,
    generate_2d_heatmap,
    generate_2d_point_mask,
    z_slice_normal_dist_3d,
)
from yagm.transforms.keypoints.helper import kpts_pad_for_batching, kpts_spatial_crop
from yagm.transforms.sliding_window import get_sliding_patch_positions
from yagm.transforms.tta_3d import build_tta

from ecg.utils import constants
from ecg.utils import misc as misc_utils

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

logger = logging.getLogger(__name__)


CV2_INTERPOLATION_METHODS = {
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "bilinear": cv2.INTER_LINEAR,
    "trilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}
NUM_GRID_ROWS, NUM_GRID_COLS = 43, 55
NUM_GRID_POINTS = NUM_GRID_ROWS * NUM_GRID_COLS
NUM_MAIN_POINTS = 57
NUM_KPTS = 2422
assert NUM_MAIN_POINTS + NUM_GRID_POINTS == NUM_KPTS
KPT_CLASSES = [0] * NUM_GRID_POINTS + list(range(1, NUM_MAIN_POINTS + 1))
KPT_CLASSES_ARR = np.array(KPT_CLASSES)[..., None]  # (N, 1)

DUP = 4
OVERSAMPLE_FACTORS = {
    1: 1,
    3: DUP,
    4: DUP,
    5: DUP,
    6: DUP,
    9: DUP,
    10: DUP,
    11: DUP,
    12: DUP,
}
AUG_MAX_ABS_ANGLE_RANGE = 30


def _ecg_get_safe_bbox(params, data, margin_xy=(100, 100)):
    """Get safe bbox to crop which contain almost all ECG leads keypoints
    Used for a custom Albumentations transform `CustomRandomSizedBBoxSafeCrop`
    """
    # img_h, img_w = params['shape'][:2]
    kpts = data["keypoints"]
    # x, y, angle, scale
    assert kpts.shape == (4, 4)

    # keep_xys = kpts[:NUM_GRID_POINTS, :2].reshape(NUM_GRID_ROWS, NUM_GRID_COLS, 2)[15:-2, 1:-2]

    # just ignore 'mms', 'mmmv'
    keep_xys = kpts[:, :2]
    min_x, min_y = keep_xys.min(axis=0).tolist()
    max_x, max_y = keep_xys.max(axis=0).tolist()
    return [
        min_x - margin_xy[0],
        min_y - margin_xy[1],
        max_x + margin_xy[0],
        max_y + margin_xy[1],
    ]


class ECGImageRotationDataset(BaseDataset):
    """
    Dataset for image rotation/angle classification/regression
    """

    def __init__(self, cfg, stage="train", cache={}):
        super().__init__(cfg, stage, cache)
        self.data_dir = cfg.env.data_dir
        self.img_dir = os.path.join(self.data_dir, "raw", "train")

        # load CV
        with open(
            os.path.join(self.data_dir, "processed", "cv", f"{cfg.cv.strategy}.json"),
            "r",
        ) as f:
            cv_meta = json.load(f)
        assert len(cv_meta["folds"]) == cfg.cv.num_folds
        logger.info(
            "Loaded fold %d (total %d folds) from CV strategy %s",
            cfg.cv.fold_idx,
            cfg.cv.num_folds,
            cfg.cv.strategy,
        )
        fold_meta = cv_meta["folds"][cfg.cv.fold_idx]
        train_ids = fold_meta["train_ids"]
        val_ids = fold_meta["val_ids"]

        if stage == "train":
            if cfg.cv.train_on == "all":
                self.stage_ids = train_ids + val_ids
            elif cfg.cv.train_on == "train":
                self.stage_ids = train_ids
            else:
                raise ValueError
        else:
            if cfg.cv.val_on == "all":
                self.stage_ids = train_ids + val_ids
            elif cfg.cv.val_on == "train":
                self.stage_ids = train_ids
            elif cfg.cv.val_on == "val":
                self.stage_ids = val_ids
            else:
                raise ValueError

        # load gt dataframe
        gt_csv_path = os.path.join(
            self.data_dir, "processed", f"{cfg.data.gt_name}.csv"
        )
        # .filter() keep original order -> okay
        gt_df = (
            pl.scan_csv(gt_csv_path)
            .with_row_index("index")
            .select(pl.col("index", "id", "type_id", "rot_code", "H"))
            .filter(pl.col("id").is_in(self.stage_ids))
            .with_columns(
                pl.col("H")
                .map_elements(
                    lambda x: (
                        misc_utils.get_rotation_angle_from_homo_mat(np.array(eval(x)))
                    ),
                    return_dtype=pl.Float32,
                )
                .alias("angle")
            )
            .collect()
        )
        assert gt_df["id"].n_unique() == len(self.stage_ids)
        logger.info(
            "Loaded %s dataframe with shape %s (%d ids):\n%s",
            stage,
            gt_df.shape,
            len(self.stage_ids),
            gt_df,
        )

        # load numpy array of XY keypoints
        # we can load them from csv file gt_csv_path, but a separate npy file is much faster
        stage_idxs = gt_df["index"].to_numpy()
        gt_kpt_xys = np.load(
            os.path.join(self.data_dir, "processed", f"{cfg.data.gt_name}.npy")
        )[stage_idxs]
        assert gt_kpt_xys.shape == (len(gt_df), NUM_KPTS, 2)

        self.gt_kpt_xys = gt_kpt_xys
        self.gt_df = gt_df

        # oversample by `type_id` on train stage
        if stage == "train":
            oversample_idxs = []
            for i, row in enumerate(gt_df.iter_rows(named=True)):
                oversample_factor = OVERSAMPLE_FACTORS[row["type_id"]]
                assert oversample_factor > 0
                oversample_idxs.extend([i] * oversample_factor)
            logger.info(
                "Oversample %s stage from %d to %d samples",
                stage,
                len(gt_df),
                len(oversample_idxs),
            )
            self.oversample_idxs = oversample_idxs

        # BUILD TRANSFORM FUNC
        self.transform, augment = self.build_transform()
        if stage == "train":
            self.augment = augment
        logger.info("%s transform:\n%s", self.stage, self.transform)

    # @property
    # def worker_init_fn(self):
    #     def _woker_init_fn(worker_id: int, rank=None):
    #         worker_info = torch.utils.data.get_worker_info()
    #         dataset = worker_info.dataset
    #         logger.debug(
    #             "Worker %d/%d: set MONAI transform random state to %d",
    #             worker_info.id + 1,
    #             worker_info.num_workers,
    #             worker_info.seed,
    #         )
    #         assert id(dataset) == id(self)

    #         # for val, set pytorch's number of threads to 1
    #         # otherwise, it's stuck..
    #         # ref: https://github.com/pytorch/pytorch/issues/89693
    #         if self.stage != "train":
    #             # print('{self.stage} OLD NUM THREADS:', torch.get_num_threads())
    #             torch.set_num_threads(1)

    #     return _woker_init_fn

    def __len__(self):
        if self.stage == "train":
            return len(self.oversample_idxs) * 4
        else:
            # return 500
            # @TODO - TTA for validation (e.g, more rotation angles)
            return len(self.gt_df) * 4

    def _pull_item(self, idx, mix=False):
        assert not mix
        dst_rotation_label = idx % 4
        oversample_idx = idx // 4
        if self.stage == "train":
            sample_idx = self.oversample_idxs[oversample_idx]
        else:
            sample_idx = oversample_idx
            # sample_idx = random.randint(0, len(self.gt_df))
            # print(sample_idx)

        # load image
        sample = self.gt_df[sample_idx].to_dicts()[0]

        sample_id = sample["id"]
        type_id = sample["type_id"]
        rot_code = sample["rot_code"]
        # H is used to transform current image TO reference image
        if sample["H"] is None:
            # type_id == 0001
            assert type_id == 1

        ### LOAD RGB IMAGE
        img_path = os.path.join(
            self.img_dir, str(sample_id), f"{sample_id}-{type_id:04d}.png"
        )
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # correct the orientation
        if rot_code is not None:
            assert rot_code == int(rot_code)
            rot_code = int(rot_code)
            img = cv2.rotate(img, rot_code)

        if self.stage != "train":
            if sample["angle"] is None:
                assert type_id == 1
                cur_angle = 0
            else:
                cur_angle = sample["angle"]
            img = self.transform(image=img)["image"]
            cur_rotation_idx = misc_utils.angle_to_rotation_idx(cur_angle)
            # need to rotate this_count * 90 degree clockwise to match dst rotation_idx
            need_rot90_count = (dst_rotation_label - cur_rotation_idx) % 4
            need_rot90_code = [
                None,
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_180,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            ][need_rot90_count]
            if need_rot90_code is not None:
                img = cv2.rotate(img, need_rot90_code)
            dst_angle = cur_angle + 90 * need_rot90_count
        else:
            # FOR TRAIN STAGE
            # albumentation keypoint: (x,y,scale), we use original scale as 1.0
            src_corners = np.ones((4, 3), dtype=np.float32)
            ori_kpt_xys = self.gt_kpt_xys[sample_idx]
            assert ori_kpt_xys.shape == (NUM_KPTS, 2)

            # Detectron2 adds 0.5 to COCO keypoint coordinates to convert them
            # from discrete pixel indices to floating point coordinates
            # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
            # but in this case, keypoints are already in continous coordinate space

            # ref_kpt_names.index('dc_0_0'), ref_kpt_names.index('s_0_2_t'), ref_kpt_names.index('s_2_2_b'), ref_kpt_names.index('dc_3_3')
            # [2391, 2369, 2386, 2406]
            src_corners[:4, :2] = ori_kpt_xys[[2391, 2369, 2386, 2406]]
            tdata = self.augment(image=img, keypoints=src_corners)
            img = tdata["image"]
            src_corners = tdata["keypoints"]
            assert src_corners.shape == (4, 3)  # x, y, scale
            src_corners = src_corners[:, :2]  # xy

            # calculate the transformation matrix M mapping transformed coordinates to original coordinates
            ref_corners = constants.REF_KPT_XYS[[2391, 2369, 2386, 2406]]

            # M is transformation matrix from CURENT IMAGE to REFERENCE (TEMPLATE) IMAGE
            M = cv2.getPerspectiveTransform(
                src=ref_corners[None], dst=src_corners[None, ...]
            )  # (3,3)

            cur_angle = misc_utils.get_rotation_angle_from_homo_mat(M)
            cur_rotation_idx = misc_utils.angle_to_rotation_idx(cur_angle)

            if (
                abs(misc_utils.angular_diff(cur_angle, 90 * cur_rotation_idx))
                > AUG_MAX_ABS_ANGLE_RANGE
            ) or (random.random() < self.cfg.data.force_match_rand_angle_prob):
                dst_angle = random.uniform(
                    90 * dst_rotation_label - AUG_MAX_ABS_ANGLE_RANGE,
                    90 * dst_rotation_label + AUG_MAX_ABS_ANGLE_RANGE,
                )
                need_to_rotate_angle = dst_angle - cur_angle
                img = misc_utils.rotate_image(
                    img,
                    -need_to_rotate_angle,
                    border_mode=cv2.BORDER_CONSTANT,
                    border_val=(128, 128, 128),
                )
            else:
                # need to rotate this_count * 90 degree clockwise to match dst rotation_idx
                need_rot90_count = (dst_rotation_label - cur_rotation_idx) % 4
                need_rot90_code = [
                    None,
                    cv2.ROTATE_90_CLOCKWISE,
                    cv2.ROTATE_180,
                    cv2.ROTATE_90_COUNTERCLOCKWISE,
                ][need_rot90_count]
                if need_rot90_code is not None:
                    img = cv2.rotate(img, need_rot90_code)
                dst_angle = cur_angle + 90 * need_rot90_count

            img = self.transform(image=img)["image"]

        dst_angle %= 360
        rad = math.radians(dst_angle)
        angle_sine = math.sin(rad)
        angle_cosine = math.cos(rad)

        ret = {
            "idx": torch.tensor(sample_idx, dtype=torch.long),
            "image": torch.from_numpy(img.transpose(2, 0, 1)),  # CHW
            "angle": torch.tensor(dst_angle, dtype=torch.float32),
            "sine_cosine": torch.tensor(
                [angle_sine, angle_cosine], dtype=torch.float32
            ),
            "label": torch.tensor(dst_rotation_label, dtype=torch.long),
        }

        # debug
        if 0:
            sample = ret
            img = sample["image"].permute(1, 2, 0).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            label = sample["label"].item()
            angle = sample["angle"].item()

            save_path = os.path.join(
                f"tmp/{label}/{sample_id}_type{type_id}_rot{rot_code}_angle_{angle}.jpg"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)

        return ret

    def __getitem__(self, idx):
        return self._pull_item(idx, mix=False)

    def build_transform(self):
        acfg = self.cfg.data.aug
        # BUILD TRANSFORM
        keypoint_params = A.KeypointParams(
            format="xys",
            remove_invisible=False,
            check_each_transform=False,
        )
        compose_kwargs = dict(keypoint_params=keypoint_params)
        if acfg.keep_ratio:
            resize_op = AC.LongestMaxHW(
                self.cfg.data.img_size,
                CV2_INTERPOLATION_METHODS[self.cfg.data.interp],
                p=1.0,
            )
        else:
            resize_op = A.Resize(
                *self.cfg.data.img_size,
                interpolation=CV2_INTERPOLATION_METHODS[self.cfg.data.interp],
                p=1.0,
            )
        transform = A.Compose(
            [
                resize_op,
                A.PadIfNeeded(
                    self.cfg.data.img_size[0],
                    self.cfg.data.img_size[1],
                    position="center",  # important to protect near-boundary consistency
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(128, 128, 128),
                    p=1.0,
                ),
            ],
            p=1.0,
            **compose_kwargs,
        )

        augment = A.Compose(
            [
                # FLIP
                # A.HorizontalFlip(p=acfg.hflip_p),
                # A.VerticalFlip(p=acfg.vflip_p),
                # GEOMETRIC
                A.OneOf(
                    [
                        # strong rotate
                        A.Affine(
                            scale={"x": acfg.affine_scale, "y": acfg.affine_scale},
                            translate_percent=None,
                            rotate=None,
                            shear={"x": (-15, 15), "y": (-15, 15)},
                            interpolation=CV2_INTERPOLATION_METHODS[acfg.interp],
                            cval=(128, 128, 128),
                            mode=cv2.BORDER_CONSTANT,
                            fit_output=acfg.affine_fit_output,
                            keep_ratio=acfg.keep_ratio,
                            balanced_scale=True,
                            p=0.4,
                        ),
                        A.Perspective(
                            scale=(0.04, 0.08),
                            keep_size=False,
                            pad_mode=cv2.BORDER_CONSTANT,
                            pad_val=(128, 128, 128),
                            fit_output=acfg.affine_fit_output,
                            interpolation=CV2_INTERPOLATION_METHODS[acfg.interp],
                            p=0.3,
                        ),
                        # very slow + in-accurate (approximate) keypoints
                        # how to deal with out-of-image keypoints?
                        # this break intrinsic contrains -> not well-combined with other Dropout augs
                        # A.OneOf([
                        #     A.PiecewiseAffine(scale=(0.025, 0.025), nb_rows=4, nb_cols=4, interpolation=cv2.INTER_LINEAR, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, keypoints_threshold=0.01, p=0.33),
                        #     A.PiecewiseAffine(scale=(0.016, 0.016), nb_rows=6, nb_cols=6, interpolation=cv2.INTER_LINEAR, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, keypoints_threshold=0.01, p=0.33),
                        #     A.PiecewiseAffine(scale=(0.0125, 0.0125), nb_rows=8, nb_cols=8, interpolation=cv2.INTER_LINEAR, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, keypoints_threshold=0.01, p=0.34)
                        # ], p=0.1)
                    ],
                    p=acfg.affine_p,
                ),
                # RANDOM CROP
                # crop go after geometric to prevent too much information loss
                # if random crop is applied, it "SHOULD" returned an image match the model's image size ASPECT_RATIO
                # so that no black constant padding is needed, which confuse the keypoint labels
                # e.g, some keypoint is not visible on padding area, but still included in image so kptness_mask still True
                A.OneOf(
                    [
                        # crop, keep all lead bboxes
                        AC.CustomRandomSizedBBoxSafeCrop(
                            crop_size=None,
                            scale=(1.0, 2.0),  # unused when crop_size is not None
                            scale_relative_to="bbox",  # unused when crop_size is not None
                            ratio=(
                                self.cfg.data.img_size[1] / self.cfg.data.img_size[0],
                                self.cfg.data.img_size[1] / self.cfg.data.img_size[0],
                            ),  # width/height, unused when crop_size is not None
                            get_bbox_func=partial(
                                _ecg_get_safe_bbox, margin_xy=(10, 10)
                            ),
                            retry=10,
                            p=0.8,
                        ),
                        # just random crop without ensuring keeping lead bboxes
                        AC.CustomRandomSizedBBoxSafeCrop(
                            crop_size=None,
                            scale=(0.5, 1.0),  # unused when crop_size is not None
                            scale_relative_to="image",  # unused when crop_size is not None
                            ratio=(
                                self.cfg.data.img_size[1] / self.cfg.data.img_size[0],
                                self.cfg.data.img_size[1] / self.cfg.data.img_size[0],
                            ),  # width/height, unused when crop_size is not None
                            get_bbox_func=None,
                            retry=10,
                            p=0.2,
                        ),
                    ],
                    p=acfg.crop_p,
                ),
                # GRAYSCALE: mimic black-white scanner
                A.Compose(
                    [
                        A.OneOf(
                            [
                                A.ToGray(
                                    num_output_channels=3,
                                    method="weighted_average",
                                    p=0.8,
                                ),
                                A.ToGray(
                                    num_output_channels=3, method="from_lab", p=0.2
                                ),
                            ],
                            p=1.0,
                        ),
                        # Gray-specific intensity transforms
                        A.OneOf(
                            [
                                # wrong on float32 img
                                # A.Emboss(alpha=(0.3, 0.6), strength=(0.2, 0.8), p=0.4),
                                # A.Sharpen(alpha=(0.1, 0.3), lightness=(0.0, 0.4), p=0.5),
                                A.CLAHE(clip_limit=4.0, tile_grid_size=(16, 16), p=1.0),
                            ],
                            p=0.2,
                        ),
                    ],
                    p=0.05,
                ),
                # COLOR
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.3, 0.3),
                            contrast_limit=(-0.5, 0.5),
                            brightness_by_max=True,
                            p=0.3,
                        ),
                        A.RandomToneCurve(scale=0.3, per_channel=True, p=0.1),
                        A.RandomGamma(gamma_limit=(60, 150), p=0.1),
                        A.ColorJitter(
                            brightness=(0.7, 1.3),
                            contrast=(0.6, 1.4),
                            saturation=(0.7, 1.3),
                            hue=(-0.1, 0.1),
                            p=0.5,
                        ),
                    ],
                    p=0.1,
                ),
                # REDUCE QUALITY
                A.OneOf(
                    [
                        # @TODO - jitter on float32, implement AdaptiveDownscale based on current resolution
                        A.OneOf(
                            [
                                A.Downscale(
                                    scale_range=(0.5, 0.9),
                                    interpolation_pair={
                                        "upscale": cv2.INTER_LANCZOS4,
                                        "downscale": cv2.INTER_AREA,
                                    },
                                    p=0.1,
                                ),
                                A.Downscale(
                                    scale_range=(0.5, 0.9),
                                    interpolation_pair={
                                        "upscale": cv2.INTER_LINEAR,
                                        "downscale": cv2.INTER_AREA,
                                    },
                                    p=0.1,
                                ),
                                A.Downscale(
                                    scale_range=(0.5, 0.9),
                                    interpolation_pair={
                                        "upscale": cv2.INTER_LINEAR,
                                        "downscale": cv2.INTER_LINEAR,
                                    },
                                    p=0.8,
                                ),
                            ],
                            p=0.4,
                        ),
                        A.ImageCompression(
                            compression_type="jpeg", quality_range=(30, 90), p=0.5
                        ),
                        A.Posterize(num_bits=(4, 7), p=0.1),
                    ],
                    p=0.0,
                ),
                # BLUR
                A.OneOf(
                    [
                        # @TODO - Defocus params tunning
                        A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=0.3),
                        A.AdvancedBlur(
                            blur_limit=(3, 7),
                            sigma_x_limit=(0.2, 1.0),
                            sigma_y_limit=(0.2, 1.0),
                            rotate_limit=90,
                            beta_limit=(0.5, 8.0),
                            noise_limit=(0.9, 1.1),
                            p=0.3,
                        ),
                        A.MotionBlur(
                            blur_limit=(3, 7),
                            allow_shifted=True,
                            p=0.3,
                        ),
                        A.MedianBlur(blur_limit=(3, 5), p=0.1),
                    ],
                    p=0.0,
                ),
                # NOISE
                A.OneOf(
                    [
                        A.GaussNoise(
                            var_limit=(60, 120),
                            mean=0,
                            per_channel=True,
                            noise_scale_factor=0.5,
                            p=0.3,
                        ),
                        A.ISONoise(
                            color_shift=(0.00, 0.00),
                            intensity=(0.1, 0.4),
                            p=0.5,
                        ),
                        A.MultiplicativeNoise(
                            multiplier=(0.6, 1.4),
                            per_channel=True,
                            elementwise=True,
                            p=0.2,
                        ),
                    ],
                    p=0.0,
                ),
                # DROPOUT
                A.OneOf(
                    [
                        AC.CustomCoarseDropout(
                            fill_value=(128, 128, 128),
                            num_holes_range=(10, 20),
                            hole_height_range=(0.05, 0.1),
                            hole_width_range=(0.05, 0.1),
                            p=0.5,
                        ),
                        AC.CustomGridDropout(
                            ratio=0.2,
                            random_offset=True,
                            fill_value=(128, 128, 128),
                            holes_number_xy=((5, 10), (5, 10)),
                            p=0.3,
                        ),
                        AC.CustomXYMasking(
                            num_masks_x=(4, 8),
                            num_masks_y=(4, 8),
                            mask_x_length=(0.03, 0.05),
                            mask_y_length=(0.03, 0.05),
                            fill_value=(128, 128, 128),
                            p=0.2,
                        ),
                    ],
                    p=acfg.dropout_p,
                ),
            ],
            p=acfg.p,
            **compose_kwargs,
        )
        return transform, augment


# FOR TESTING PURPOSE
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from omegaconf import OmegaConf

    yaml_str = """

cv:
    strategy: v1/skf_1x5_rd42
    num_folds: 5
    fold_idx: 0
    train_on: train
    val_on: val

env:
    data_dir: '/home/dangnh36/datasets/ecg/'

data:
    _target_: yagm.data.base_datamodule.BaseDataModule
    _dataset_target_: ecg.data.datasets.image_rotation_dataset.ECGImageRotationDataset

    gt_name: keypoints_by_round2
    force_match_rand_angle_prob: 0.4
    img_size: [512, 512]
    interp: area

    aug:
        p: 0.6
        keep_ratio: True
        interp: ${data.interp}
        affine_p: 0.7
        affine_scale: [0.6, 1.5]
        affine_fit_output: True
        crop_p: 0.7
        dropout_p: 0.2


"""
    from lightning import seed_everything

    STAGE = "train"
    seed_everything(42)
    global_cfg = OmegaConf.create(yaml_str)
    dataset = ECGImageRotationDataset(cfg=global_cfg, stage=STAGE)

    print("DATASET LEN:", len(dataset))

    for idx in range(0, len(dataset), 1):
        sample = dataset[idx]
        print(idx, {k: [v.shape, v.dtype] for k, v in sample.items()})
