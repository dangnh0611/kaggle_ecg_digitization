import os
import sys

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
sys.path = [
    "/kaggle/working/ecg/yagm/src/",
    "/kaggle/working/ecg/src/",
    "/kaggle/working/ecg/",
    "/kaggle/working/ecg/third_party/segmentation_models_pytorch_3d",
    "/kaggle/working/ecg/third_party/slowfast",
    "/kaggle/working/ecg/third_party/timm_3d",
] + sys.path
print("SYS PATH:", sys.path, sep="\n")


import csv
import gc
import json
import logging
import math
import pickle
import random
import shutil
import time

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from yagm.transforms import albumentations_custom as AC
from yagm.transforms.keypoints.decode import decode_heatmap_2d_batched
from yagm.utils import lightning as l_utils
from yagm.utils.geometric import batch_perspective_transform_2d
# Setting up custom logging
from yagm.utils.logging import init_logging, setup_logging

from ecg.utils import constants
from ecg.utils.codecs import ColumnGaussianHeatmapCodec
from ecg.utils.crop import crop_by_global_homo, crop_by_unwrap_flow
from ecg.utils.interpolate import interp_1d_cv2
from ecg.utils.keypoint_detection_postprocess import register_grid_keypoints
from ecg.utils.misc import get_scale_xy_from_homo_mat
from ecg.utils.viz import viz_img_keypoints

### SETUP LOGGER FOR IPYTHON ###
# ref: https://github.com/ipython/ipykernel/issues/111
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Create STDERR handler
handler = logging.StreamHandler(sys.stderr)
# Create formatter and add it to the handler
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
# Set STDERR handler as the only handler
logger.handlers = [handler]


# ===================== GLOBAL CONFIG =====================
MODE = "LOCAL_VAL"

DEVICE = torch.device("cuda:0")

DO_ROTATION_INFERENCE = False
DO_DETECTION = False
DO_REMOVE_TMP_DETECTION = False
DO_REGISTER = False
DO_VIZ_DETECTION = False
DO_SIGNAL_HEATMAP_INFERENCE = False
DO_EVALUATE = True


if MODE == "LOCAL_TEST":
    TEST_ON_TOPK = None
    LOADER_NUM_WORKERS = 16
    ASSETS_DIR = "./assets"
    TEST_IMG_DIR = "/home/dangnh36/datasets/ecg/raw/test/"
    TEST_CSV_PATH = "/home/dangnh36/datasets/ecg/raw/test.csv"
    TMP_DIR = "./tmp/"
    TEMPLATE_ROOT_DIR = "/home/dangnh36/datasets/ecg/processed/templates/fs100/"
    SUBMISSION_CSV_PATH = os.path.join(TMP_DIR, "submission.csv")
elif MODE == "LOCAL_VAL":
    TEST_ON_TOPK = None
    LOADER_NUM_WORKERS = 16
    ASSETS_DIR = "./assets"
    TEST_IMG_DIR = "/home/dangnh36/datasets/ecg/raw/train/"
    # pseudo_test_fold0_1200images.csv | pseudo_test_fold0.csv | pseudo_test.csv
    TEST_CSV_PATH = (
        "/home/dangnh36/datasets/ecg/processed/pseudo_test_fold0_1200images.csv"
    )
    TMP_DIR = "./tmp/"
    TEMPLATE_ROOT_DIR = "/home/dangnh36/datasets/ecg/processed/templates/fs100/"
    SUBMISSION_CSV_PATH = os.path.join(TMP_DIR, "submission.csv")
elif MODE == "KAGGLE_TEST":
    SUBMISSION_CSV_PATH = "/kaggle/working/submission.csv"
    LOADER_NUM_WORKERS = 4
    TEST_ON_TOPK = None
    ASSETS_DIR = "/kaggle/input/ecg-checkpoints/"
    TEST_IMG_DIR = "/kaggle/input/physionet-ecg-image-digitization/test/"
    # pseudo_test_fold0_1200images.csv | pseudo_test_fold0.csv | pseudo_test.csv
    TEST_CSV_PATH = "/kaggle/input/physionet-ecg-image-digitization/test.csv"
    TMP_DIR = "/kaggle/temp/"
    TEMPLATE_ROOT_DIR = (
        "/kaggle/input/ecg-checkpoints/need_to_keep/need_to_keep/templates/fs100/"
    )
else:
    raise ValueError


# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

CV2_INTERPOLATION_METHODS = {
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "bilinear": cv2.INTER_LINEAR,
    "trilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

REF_MAIN_KPT_XYS = constants.REF_KPT_XYS[2365:]
assert REF_MAIN_KPT_XYS.shape == (57, 2)


def create_submission_file(
    predictions_dict, test_csv_path, output_path="submission.csv"
):
    """
    Streams submission to CSV using the exact 'number_of_rows'
    specified in test.csv, avoiding manual 'fs * duration' calculation.
    """
    print(f"Reading metadata from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)

    # Validation: Ensure test.csv has the expected columns
    required_cols = {"id", "lead", "number_of_rows"}
    if not required_cols.issubset(test_df.columns):
        print(
            f"⚠️ WARNING: test.csv is missing columns: {required_cols - set(test_df.columns)}"
        )
        print(
            "Falling back to manual calculation logic might be necessary if this fails."
        )

    print(f"Streaming submission to {output_path}...")

    # 1. Open CSV Writer
    # newline='' is required to prevent blank lines on Windows
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 2. Write Header
        writer.writerow(['id', 'value'])

        # 3. Iterate through test.csv rows directly
        for _, row in tqdm(
            test_df.iterrows(), total=len(test_df), desc="Processing test.csv rows"
        ):

            base_id = str(row["id"])
            lead_name = row["lead"]
            target_len = int(row["number_of_rows"])  # STRICT: Use this exact value

            # 4. Retrieve Prediction
            if base_id in predictions_dict:
                img_preds = predictions_dict[base_id]
            elif int(base_id) in predictions_dict:  # fallback if keys are ints
                img_preds = predictions_dict[int(base_id)]
            else:
                img_preds = {}  # Missing ID

            # Get specific lead signal
            if lead_name in img_preds:
                signal = img_preds[lead_name]
            else:
                signal = np.zeros(target_len, dtype=np.float32)

            # 5. Enforce Exact Length (Resample if needed)
            current_len = len(signal)

            if current_len != target_len:
                x_old = np.linspace(0, 1, current_len)
                x_new = np.linspace(0, 1, target_len)
                signal = np.interp(x_new, x_old, signal)

            # 6. Generate IDs
            # Format: {base_id}_{row_id}_{lead}
            # row_id is 0-based index
            ids = [f"{base_id}_{i}_{lead_name}" for i in range(target_len)]

            # 7. Write Chunk
            # zip() creates a lazy iterator, very memory efficient
            writer.writerows(zip(ids, signal))

    print(f"Done! CSV Submission created at {output_path}.")


def csv_to_nested_dict(csv_path):
    """
    Reads the submission.csv and reconstructs the original nested dictionary:
    { 'base_id': { 'lead': np.array([value1, value2, ...]) } }

    Uses Polars for high-performance string parsing and aggregation.
    """
    print(f"Reading {csv_path}...")

    # 1. Lazy Scan (Memory Efficient)
    # Switch from scan_parquet to scan_csv
    lf = pl.scan_csv(csv_path)

    # Regex Explanation:
    #   ^(.*)     -> Group 1: Capture Base ID (greedy, handles slashes/underscores)
    #   _(\d+)    -> Group 2: Capture Row ID (digits only, preceded by _)
    #   _([^_]+)$ -> Group 3: Capture Lead (everything after last _)

    processed_lf = lf.with_columns(
        [
            pl.col("id").str.extract(r"^(.*)_(\d+)_([^_]+)$", 1).alias("base_id"),
            pl.col("id")
            .str.extract(r"^(.*)_(\d+)_([^_]+)$", 2)
            .cast(pl.Int32)
            .alias("row_idx"),
            pl.col("id").str.extract(r"^(.*)_(\d+)_([^_]+)$", 3).alias("lead"),
        ]
    )

    # 2. Group by (Base ID, Lead) and Aggregate Values
    # We MUST sort by 'row_idx' inside the aggregation to ensure signal order is correct.
    aggregated_df = (
        processed_lf.group_by(["base_id", "lead"])
        .agg(pl.col("value").sort_by("row_idx").alias("signal"))
        .collect()
    )  # Materialize results here

    print(f"Reconstructing dictionary from {len(aggregated_df)} groupings...")

    # 3. Convert to Dictionary
    # Iterating over the aggregated groups is fast (approx 25k iterations for 1200 images * 12 leads)
    reconstructed_dict = {}

    for row in tqdm(aggregated_df.iter_rows(named=True), total=len(aggregated_df)):
        base_id = row["base_id"]
        lead = row["lead"]

        # Polars returns lists, convert back to numpy float32
        signal_array = np.array(row["signal"], dtype=np.float32)

        if base_id not in reconstructed_dict:
            reconstructed_dict[base_id] = {}

        reconstructed_dict[base_id][lead] = signal_array

    return reconstructed_dict


def load_sample_signal(img_dir, img_id):
    if "/" in img_id:
        ori_sample_id = img_id.split("/")[0]
        signal_csv_path = os.path.join(img_dir, ori_sample_id, f"{ori_sample_id}.csv")
    else:
        raise NotImplementedError
    signal_df = (
        pl.scan_csv(signal_csv_path).select(pl.col("*").cast(pl.Float64)).collect()
    )
    sig_len = len(signal_df)
    signal = signal_df.to_numpy()
    null_mask = ~np.isnan(signal)
    ret = {}

    for lead_idx, lead_name in enumerate(signal_df.columns):
        assert lead_name in constants.LEAD_TO_SIG_LEN
        start_frac, end_frac = constants.LEAD_TO_SIG_LEN[lead_name]
        valid_indices = null_mask[:, lead_idx].nonzero()[0]
        first = int(valid_indices[0])
        last = int(valid_indices[-1])
        expect_start = int(start_frac * sig_len)
        expect_end = int(end_frac * sig_len - 1)
        assert first == expect_start and last == expect_end

        lead_sig = signal[first : last + 1, lead_idx]
        assert not np.isnan(lead_sig).any()
        if lead_name == "II":
            assert len(lead_sig) == sig_len
        else:
            if sig_len % 4 == 0:
                assert len(lead_sig) == sig_len // 4
            else:
                # int: 810, ceil: 972
                assert len(lead_sig) == int(sig_len / 4) or len(lead_sig) == math.ceil(
                    sig_len / 4
                )
        ret[lead_name] = lead_sig
    return ret


def clear_dir_content(folder_path):
    if os.path.isdir(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)


class ECGImageRotationDataset(Dataset):
    def __init__(self, cfg, img_ids, img_dir):
        self.cfg = cfg
        self.img_ids = img_ids
        self.img_dir = img_dir
        print("NUMBER OF IMAGES:", len(self.img_ids))

        # BUILD TRANSFORM FUNC
        print(f"USING IMAGE SIZE={cfg.data.img_size} INTERPOLATION={cfg.data.interp}")
        self.transform = self.build_transform()
        print("Transform:\n", self.transform)

    def build_transform(self):
        acfg = self.cfg.data.aug
        # BUILD TRANSFORM
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
        )
        return transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ### LOAD RGB IMAGE
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        return {
            "idx": torch.tensor(idx, dtype=torch.long),
            "image": torch.from_numpy(img.transpose(2, 0, 1)),  # CHW
        }


class ECGKeypointDetectionDataset(Dataset):
    def __init__(self, cfg, img_ids, img_dir, rotation_map=None):
        self.cfg = cfg
        self.img_ids = img_ids
        self.img_dir = img_dir
        print("NUMBER OF IMAGES:", len(self.img_ids))
        self.rotation_map = (
            rotation_map
            if rotation_map is not None
            else {img_id: 0 for img_id in self.img_ids}
        )

        # BUILD TRANSFORM FUNC
        print(f"USING IMAGE SIZE={cfg.data.img_size} INTERPOLATION={cfg.data.interp}")
        self.transform = self.build_transform()
        print("Transform:\n", self.transform)

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
        return transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        ### LOAD RGB IMAGE
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # predicted rotation index, from 0 to 3
        # 0: 0 degree, 1: 90 degree, 2: 180 degree, 3: 270 degree
        rot_code = self.rotation_map[img_id]
        # after using this model to predict rotation index
        # standardize the orientation of 0 degree
        std_rotation_code = [
            None,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_CLOCKWISE,
        ][rot_code]
        if std_rotation_code is not None:
            img = cv2.rotate(img, std_rotation_code)
        else:
            # 0 to 0, unchanged
            pass

        # 4 corners of original image
        ori_h, ori_w = img.shape[:2]
        src_corners = np.array(
            [[0, 0], [ori_w, 0], [ori_w, ori_h], [0, ori_h]], dtype=np.float32
        )  # xy

        # albumentation keypoint: (x,y,scale), we use original scale as 1.0
        albu_keypoints = np.ones((4, 3), dtype=np.float32)
        albu_keypoints[:, :2] = src_corners

        tdata = self.transform(image=img, keypoints=albu_keypoints)
        img = tdata["image"]
        albu_keypoints = tdata["keypoints"]
        assert albu_keypoints.shape == (4, 3)
        # calculate the transformation matrix M mapping transformed coordinates to original coordinates
        dst_corners = albu_keypoints[:, :2]  # xy
        M = cv2.getPerspectiveTransform(
            src=dst_corners[None], dst=src_corners[None, ...]
        )  # (3,3)

        return {
            "idx": torch.tensor(idx, dtype=torch.long),
            "image": torch.from_numpy(img.transpose(2, 0, 1)),  # CHW
            "ori_wh": torch.tensor([ori_w, ori_h], dtype=torch.float32),
            "M": torch.from_numpy(M),
        }


class ECGSignalHeatmapDataset(Dataset):
    def __init__(self, cfg, img_ids, img_dir, template_root_dir, rotation_map=None):
        self.cfg = cfg
        self.img_ids = img_ids
        self.img_dir = img_dir
        print("NUMBER OF IMAGES:", len(self.img_ids))
        self.rotation_map = (
            rotation_map
            if rotation_map is not None
            else {img_id: 0 for img_id in self.img_ids}
        )

        # infer some configs
        self.GT_H, self.GT_W, self.GT_L = self.cfg.data.heatmap_hwl
        self.IMG_H = round(self.GT_H * self.cfg.data.heatmap_stride[0])
        self.IMG_W = round(self.GT_W * self.cfg.data.heatmap_stride[1])
        self.img_size = [self.IMG_H, self.IMG_W]
        # signal cover exactly 500 pixels width in groundtruth mask (width=512 pixels)
        # 503.93700787401576 = 492.12598425196853 / (500 / 512)
        self.CROP_W = 492.12598425196853 / (self.GT_L / self.GT_W)
        if cfg.data.keep_aspect_ratio:
            _crop_h = self.CROP_W * self.IMG_H / self.IMG_W
            if cfg.data.crop_h is None:
                self.CROP_H = _crop_h
            else:
                assert abs(cfg.data.crop_h - _crop_h) < 1e-6
                self.CROP_H = cfg.data.crop_h
        else:
            assert cfg.data.crop_h is not None
            self.CROP_H = cfg.data.crop_h
        self.MAX_ABS_SIGNAL = self.CROP_H / 2 / constants.REF_Y_UNIT_PIXELS
        logger.info(
            "GT_H=%d GT_W=%d GT_L=%d IMG_H=%d IMG_W=%d CROP_H=%f CROP_W=%f",
            self.GT_H,
            self.GT_W,
            self.GT_L,
            self.IMG_H,
            self.IMG_W,
            self.CROP_H,
            self.CROP_W,
        )
        logger.info(
            "Current codec allow to encode/decode signal in range %s",
            [-self.MAX_ABS_SIGNAL, self.MAX_ABS_SIGNAL],
        )

        self.crop_names = [
            "I",
            "II_short",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "II_long_0",
            "II_long_1",
            "II_long_2",
            "II_long_3",
        ]
        self.samples = []
        for img_idx, img_id in enumerate(img_ids):
            fs = EXPECTED_GT_FS[img_id]
            for crop_name in self.crop_names:
                if crop_name in ["II_short", "II_long_0", "II_long_2"]:
                    len10 = EXPECTED_GT_LEN[img_id]["II"]
                    # 10250 // 4 = 2562
                    gt_len = len10 // 4
                elif crop_name in ["II_long_1", "II_long_3"]:
                    len10 = EXPECTED_GT_LEN[img_id]["II"]
                    # 10250 // 4 + 1 = 2563
                    gt_len = len10 // 2 - len10 // 4
                else:
                    gt_len = EXPECTED_GT_LEN[img_id][crop_name]
                self.samples.append([img_idx, img_id, crop_name, gt_len, fs])

        print("NUMBER OF SAMPLES:", len(self.samples))

        # LOAD TEMPLATE
        if cfg.data.use_template is not None:
            template_dir = os.path.join(
                template_root_dir,
                f"{self.IMG_H}x{self.IMG_W}",
            )
            _template_names = os.listdir(template_dir)
            self.cached_template_imgs = {}
            for _template_name in os.listdir(template_dir):
                template_img = cv2.imread(os.path.join(template_dir, _template_name))
                if cfg.data.use_template == "rgb":
                    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
                elif cfg.data.use_template == "gray":
                    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)[
                        ..., None
                    ]
                elif cfg.data.use_template == "red":
                    # take R from BGR -> (H, W, 1)
                    template_img = template_img[:, :, 2:3]
                else:
                    raise ValueError
                assert template_img.ndim == 3 and template_img.shape[:2] == (
                    self.IMG_H,
                    self.IMG_W,
                )
                self.cached_template_imgs[_template_name.replace(".png", "")] = (
                    template_img
                )
            logger.info(
                "Loaded %d template images: %s",
                len(self.cached_template_imgs),
                list(self.cached_template_imgs.keys()),
            )

        # load numpy array of XY keypoints
        self.detected_kpt_xys = np.load(DETECTION_NPY_SAVE_PATH)
        assert self.detected_kpt_xys.shape == (len(img_ids), constants.NUM_KPTS, 2)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, img_id, crop_name, gt_len, fs = self.samples[idx]
        detected_keypoints = self.detected_kpt_xys[img_idx]
        assert detected_keypoints.shape == (constants.NUM_KPTS, 2)

        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # predicted rotation index, from 0 to 3
        # 0: 0 degree, 1: 90 degree, 2: 180 degree, 3: 270 degree
        rot_code = self.rotation_map[img_id]
        # after using this model to predict rotation index
        # standardize the orientation of 0 degree
        std_rotation_code = [
            None,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_CLOCKWISE,
        ][rot_code]
        if std_rotation_code is not None:
            img = cv2.rotate(img, std_rotation_code)
        else:
            # 0 to 0, unchanged
            pass

        # CROP BY WARP
        if self.cfg.data.warp.method == "global_homo":
            crop = crop_by_global_homo(
                crop_name,
                img,
                detected_keypoints,
                signal_length_secs=self.cfg.data.signal_length_secs,
                warp_h=self.IMG_H,
                warp_w=self.IMG_W,
                crop_h=self.CROP_H,
                crop_w=self.CROP_W,
                filter_nearby=self.cfg.data.warp.global_homo.filter_nearby,
                interpolation_mode=CV2_INTERPOLATION_METHODS[self.cfg.data.warp.interp],
            )
        elif self.cfg.data.warp.method == "unwrap_flow":
            # -------------------------------------------
            # dirty hack to quickly obtain TEMPLATE CROPS
            # DON'T COMMENT OUT
            # img = cv2.imread('/home/dangnh36/datasets/ecg/processed/REFERENCE_TEMPLATE_FS100.png')
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # augmented_detected_keypoints = constants.REF_KPT_XYS.copy()
            # -------------------------------------------

            try:
                crop = crop_by_unwrap_flow(
                    crop_name,
                    img,
                    detected_keypoints=detected_keypoints,
                    signal_length_secs=self.cfg.data.signal_length_secs,
                    warp_h=self.IMG_H,
                    warp_w=self.IMG_W,
                    crop_h=self.CROP_H,
                    crop_w=self.CROP_W,
                    filter_inside=self.cfg.data.warp.unwrap_flow.filter_inside,
                    unwrap_method=self.cfg.data.warp.unwrap_flow.method,
                    interpolation=self.cfg.data.warp.interp,
                    warp_backend=self.cfg.data.warp.backend,
                    warp_device=self.cfg.data.warp.device,
                    flow_smooth_sigma=self.cfg.data.warp.unwrap_flow.smooth_sigma,
                    _shift_0dot5_in_cv2_remap=self.cfg.data.warp.unwrap_flow._shift_0dot5_in_cv2_remap,
                    **self.cfg.data.warp.unwrap_flow.kwargs,
                )
            except:
                print("EXCEPTION OCCUR, CAN NOT CROP BY WARP FLOW")
                print('FALLBACK TO GLOBAL HOMO')
                try:
                    crop = crop_by_global_homo(
                        crop_name,
                        img,
                        detected_keypoints,
                        signal_length_secs=self.cfg.data.signal_length_secs,
                        warp_h=self.IMG_H,
                        warp_w=self.IMG_W,
                        crop_h=self.CROP_H,
                        crop_w=self.CROP_W,
                        filter_nearby=self.cfg.data.warp.global_homo.filter_nearby,
                        interpolation_mode=CV2_INTERPOLATION_METHODS[self.cfg.data.warp.interp],
                    )
                except:
                    # SAFETY ONLY :) YOLO
                    crop = img[:self.img_H, :self.img_W]


            # -----------------------------------------------
            # DON'T COMMENT OUT
            # template_crop_save_path = f'/home/dangnh36/datasets/ecg/processed/templates/fs100/{self.IMG_H}x{self.IMG_W}/{crop_name}.png'
            # os.makedirs(os.path.dirname(template_crop_save_path), exist_ok=True)
            # cv2.imwrite(template_crop_save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            # -----------------------------------------------
        else:
            raise ValueError

        # concat with empty template if needed
        if self.cfg.data.use_template is not None:
            template = self.cached_template_imgs[crop_name]
            # usually (H, W, 6) or (H, W, 4)
            crop = np.concatenate([crop, template], axis=-1)

        return {
            "idx": torch.tensor(idx, dtype=torch.long),
            "image": torch.from_numpy(crop.transpose(2, 0, 1)),  # CHW
        }


# =========== LOAD METADATA ============
# Load CSV using Polars for speed
TEST_DF = pl.read_csv(TEST_CSV_PATH)
if TEST_ON_TOPK is not None and MODE != "KAGGLE_TEST":
    TEST_DF = TEST_DF[: TEST_ON_TOPK * 12]
# F**K, TEST_DF['id'].unique().to_list() is not deterministic
ALL_IMG_IDS = sorted(list(set(TEST_DF["id"].cast(pl.String).to_list())))
print("NUMBER OF TEST IDS:", len(ALL_IMG_IDS))
print("FIRST 5 IMAGE IDS:", ALL_IMG_IDS[:5])

EXPECTED_GT_LEN = {}
EXPECTED_GT_FS = {}
for row in TEST_DF.iter_rows(named=True):
    EXPECTED_GT_LEN.setdefault(str(row["id"]), {})[row["lead"]] = row["number_of_rows"]
    EXPECTED_GT_FS[str(row["id"])] = row["fs"]


# ======================= FACE ROTATION ========================

IMAGE_ROTATION_CONFIG_PATH = os.path.join(ASSETS_DIR, "ROTATION_EFFL3_512.yaml")
IMAGE_ROTATION_CHECKPOINT_PATH = os.path.join(
    ASSETS_DIR,
    "ROTATION_EFFL3_512_ep=6_step=10003_val_accuracy=1.ckpt",
)


# IMAGE_ROTATION_CONFIG_PATH = os.path.join(ASSETS_DIR, "ROTATE_EFFVIT_B2_512_config.yaml")
# IMAGE_ROTATION_CHECKPOINT_PATH = os.path.join(
#     ASSETS_DIR,
#     "ROTATE_EFFVIT_B2_512_ep11_step12005_val_accuracy1.000000_val_angle_mae1.103764.ckpt",
# )


IMAGE_ROTATION_CONFIG = OmegaConf.load(IMAGE_ROTATION_CONFIG_PATH)

# !!! overwrite some configs !!!
IMAGE_ROTATION_CONFIG.model.encoder.pretrained = False

ROTATION_TMP_DIR = os.path.join(TMP_DIR, "rotation")
os.makedirs(ROTATION_TMP_DIR, exist_ok=True)
ROTATION_MAP_PATH = os.path.join(ROTATION_TMP_DIR, "rotation_map.json")


def image_rotation_inference():
    global IMAGE_ROTATION_CONFIG
    image_rotation_dataset = ECGImageRotationDataset(
        IMAGE_ROTATION_CONFIG, img_ids=ALL_IMG_IDS, img_dir=TEST_IMG_DIR
    )
    image_rotation_loader = DataLoader(
        image_rotation_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=LOADER_NUM_WORKERS,
        drop_last=False,
        pin_memory=False,
    )
    # load model
    image_rotation_task = l_utils.build_task(IMAGE_ROTATION_CONFIG)

    l_utils.load_lightning_state_dict(
        model=image_rotation_task,
        ckpt_path=IMAGE_ROTATION_CHECKPOINT_PATH,
        cfg=IMAGE_ROTATION_CONFIG,
    )
    logger.info("Loaded Pytorch state dict from %s", IMAGE_ROTATION_CHECKPOINT_PATH)
    model = image_rotation_task.model
    del image_rotation_task
    gc.collect()
    model.to(DEVICE).eval()
    # print(model)

    # PERFORM INFERENCE PER-FOLD
    all_pred_cls = []
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        for batch_idx, batch in tqdm(
            enumerate(image_rotation_loader),
            total=len(image_rotation_loader),
        ):
            # print(batch_idx, [(k, v.shape) for k, v in batch.items()])
            batch_imgs = batch["image"].to(DEVICE)
            pred_cls_logit, _pred_reg_logit = model(batch_imgs)
            # (N, 4) -> (N,)
            pred_cls = torch.argmax(pred_cls_logit, dim=1)
            all_pred_cls.append(pred_cls.cpu().numpy())

    all_pred_cls = np.concatenate(all_pred_cls, axis=0).tolist()
    assert len(all_pred_cls) == len(ALL_IMG_IDS)

    rotation_map = {}
    for img_id, rotation_idx in zip(ALL_IMG_IDS, all_pred_cls):
        rotation_map[img_id] = rotation_idx
        if rotation_idx != 0:
            print("WRONG ROTATION:", img_id, "-->", rotation_idx)

    with open(ROTATION_MAP_PATH, "w") as f:
        json.dump(rotation_map, f)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rotation_map


if DO_ROTATION_INFERENCE:
    ROTATION_MAP = image_rotation_inference()
else:
    with open(ROTATION_MAP_PATH, "r") as f:
        ROTATION_MAP = json.load(f)


# ========== KEYPOINTS DETECTION STAGE ==========
DETECTION_TMP_DIR = os.path.join(TMP_DIR, "detection")
os.makedirs(DETECTION_TMP_DIR, exist_ok=True)

KEYPOINT_DETECTION_CONFIG_PATH = os.path.join(
    ASSETS_DIR, "KEYPOINT_DETECTION_ROUND2_DSNT_config.yaml"
)
KEYPOINT_DETECTION_CONFIG = OmegaConf.load(KEYPOINT_DETECTION_CONFIG_PATH)
DETECTION_NPY_SAVE_PATH = os.path.join(DETECTION_TMP_DIR, "all_predicted_keypoints.npy")


# !!! overwrite some configs !!!
KEYPOINT_DETECTION_CONFIG.model.encoder.pretrained = False
KEYPOINT_DETECTION_CONFIG.model.change_stem_stride = None

# print("KEYPOINT DETECTION CONFIG:", KEYPOINT_DETECTION_CONFIG, sep="\n")


def keypoint_detection_inference():
    global KEYPOINT_DETECTION_CONFIG, ROTATION_MAP
    keypoint_detection_dataset = ECGKeypointDetectionDataset(
        KEYPOINT_DETECTION_CONFIG,
        img_ids=ALL_IMG_IDS,
        img_dir=TEST_IMG_DIR,
        rotation_map=ROTATION_MAP,
    )
    keypoint_detection_loader = DataLoader(
        keypoint_detection_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=LOADER_NUM_WORKERS,
        drop_last=False,
        pin_memory=False,
    )
    keypoint_detection_input_wh = (
        torch.tensor(list(KEYPOINT_DETECTION_CONFIG.data.img_size))
        .reshape(1, 1, 2)
        .contiguous()
        .to(DEVICE)
    )

    FOLD_CKPT_NAMES = [
        "KEYPOINT_DETECTION_ROUND2_DSNT_FOLD0_ep0_step2000_val_best__AVG_ACC0.999509.ckpt",
        "KEYPOINT_DETECTION_ROUND2_DSNT_FOLD1_ep1_step4000_val_best__AVG_ACC0.999206.ckpt",
        "KEYPOINT_DETECTION_ROUND2_DSNT_FOLD2_ep0_step2000_val_best__AVG_ACC0.999356.ckpt",
        "KEYPOINT_DETECTION_ROUND2_DSNT_FOLD3_ep0_step2000_val_best__AVG_ACC0.999238.ckpt",
        "KEYPOINT_DETECTION_ROUND2_DSNT_FOLD4_ep0_step3000_val_best__AVG_ACC0.999585.ckpt",
    ]
    ACTIVE_FOLD_IDS = [0, 1, 2, 3, 4]

    # LOAD MODELS
    for fold_idx, fold_id in enumerate(ACTIVE_FOLD_IDS):
        fold_ckpt_name = FOLD_CKPT_NAMES[fold_idx]
        fold_ckpt_path = os.path.join(ASSETS_DIR, fold_ckpt_name)
        keypoint_detection_task = l_utils.build_task(KEYPOINT_DETECTION_CONFIG)
        l_utils.load_lightning_state_dict(
            model=keypoint_detection_task,
            ckpt_path=fold_ckpt_path,
            cfg=KEYPOINT_DETECTION_CONFIG,
        )
        logger.info("Loaded Pytorch state dict from %s", fold_ckpt_path)
        fold_model = keypoint_detection_task.model
        fold_model.to(DEVICE).eval()
        # print(fold_model)

        # FOLD_DETECTION_TMP_DIR = os.path.join(DETECTION_TMP_DIR, f'fold_{fold_idx}')
        # os.makedirs(FOLD_DETECTION_TMP_DIR, exist_ok = True)

        # PERFORM INFERENCE PER-FOLD
        count = -1
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            for batch_idx, batch in tqdm(
                enumerate(keypoint_detection_loader),
                total=len(keypoint_detection_loader),
            ):
                # print(batch_idx, [(k, v.shape) for k, v in batch.items()])
                batch_imgs = batch["image"].to(DEVICE)
                B = batch_imgs.shape[0]
                # (N, 58, 2048, 2048), (N, 57, 2)
                heatmap_all_pred, dsnt_main_kpt_pred, _reg_main_kpt_pred = fold_model(
                    batch_imgs
                )
                assert _reg_main_kpt_pred is None
                # (N, 2048, 2048)
                heatmap_grid_pred = (
                    F.sigmoid(heatmap_all_pred[:, 0]).half().cpu().numpy()
                )
                # from [0, 1] to [0, W] and [0, H]
                dsnt_main_kpt_xy_pixels_pred = (
                    dsnt_main_kpt_pred * keypoint_detection_input_wh
                )
                dsnt_main_kpt_xy_pixels_pred = (
                    dsnt_main_kpt_xy_pixels_pred.cpu().numpy()
                )
                batch_M = batch["M"].numpy()

                # from model input (2048x2048) image space, transform back to original image space (original resolution)
                dsnt_main_kpt_xy_pixels_pred = batch_perspective_transform_2d(
                    dsnt_main_kpt_xy_pixels_pred,  # (N, L, 2)
                    batch_M,  # (N, 3, 3)
                )

                # print('OUTPUT:', heatmap_all_pred.shape, heatmap_grid_pred.shape, heatmap_grid_pred.dtype,
                #     dsnt_main_kpt_xy_pixels_pred.shape, dsnt_main_kpt_xy_pixels_pred.dtype)

                # gather the main keypoint's confident score
                # left for future work :)

                # N, C, H, W = heatmap.shape
                # indices = dsnt_main_kpt_xy_pixels_pred
                # x = indices[:, :, 0].long()
                # y = indices[:, :, 1].long()
                # x = x.clamp(0, W - 1)
                # y = y.clamp(0, H - 1)
                # # Create auxiliary indices for Batch and Channel dimensions
                # # batch_idx: Shape (N, 1) -> Broadcasts to (N, C)
                # batch_idx = torch.arange(N, device=heatmap.device).unsqueeze(1)
                # # channel_idx: Shape (1, C) -> Broadcasts to (N, C)
                # channel_idx = torch.arange(C, device=heatmap.device).unsqueeze(0)
                # # Use Advanced Indexing
                # conf_values = heatmap[batch_idx, channel_idx, y, x]

                # SAVE GRID HEATMAP TO NPY FILE
                batch_idxs = batch["idx"].tolist()
                for batch_element_idx in range(B):
                    count += 1
                    img_idx = batch_idxs[batch_element_idx]
                    assert img_idx == count
                    img_id = ALL_IMG_IDS[img_idx]
                    if "VAL" in MODE:
                        save_heatmap_npz_path = os.path.join(
                            DETECTION_TMP_DIR, f'{img_id.replace("/", "--")}.npz'
                        )
                    else:
                        save_heatmap_npz_path = os.path.join(
                            DETECTION_TMP_DIR, f"{img_id}.npz"
                        )
                    save_heatmap_grid = heatmap_grid_pred[batch_element_idx]
                    save_dsnt_main_kpt_xy_pixels = dsnt_main_kpt_xy_pixels_pred[
                        batch_element_idx
                    ]
                    M = batch_M[batch_element_idx]
                    ori_wh = batch["ori_wh"][batch_element_idx].numpy()

                    if fold_idx == 0:
                        # (2048, 2048)
                        # save_heatmap_grid = save_heatmap_grid
                        # (1, 57, 2)
                        save_dsnt_main_xy = save_dsnt_main_kpt_xy_pixels[None]
                    else:
                        npzfile = np.load(save_heatmap_npz_path)
                        prev_avg_heatmap_grid = npzfile["heatmap_grid"]
                        prev_dsnt_main_xy = npzfile["dsnt_main_xy"]
                        prev_M = npzfile["M"]
                        prev_ori_wh = npzfile["ori_wh"]
                        assert np.array_equal(prev_M, M) and np.array_equal(
                            prev_ori_wh, ori_wh
                        )
                        # (2048, 2048)
                        save_heatmap_grid = (
                            prev_avg_heatmap_grid * fold_idx + save_heatmap_grid
                        ) / (fold_idx + 1)
                        # (NUM_FOLDS, 57, 2)
                        save_dsnt_main_xy = np.concatenate(
                            [prev_dsnt_main_xy, save_dsnt_main_kpt_xy_pixels[None]],
                            axis=0,
                        )

                    print(
                        f"FOLD_IDX={fold_idx} FOLD_ID={fold_id} HEATMAP={save_heatmap_grid.shape} DSNT={save_dsnt_main_xy.shape}"
                    )
                    np.savez(
                        save_heatmap_npz_path,
                        heatmap_grid=save_heatmap_grid,
                        dsnt_main_xy=save_dsnt_main_xy,
                        M=M,
                        ori_wh=ori_wh,
                    )

        del fold_model
        gc.collect()
        torch.cuda.empty_cache()

    del keypoint_detection_dataset, keypoint_detection_loader
    del keypoint_detection_input_wh
    gc.collect()
    torch.cuda.empty_cache()
    print("DONE ")


if DO_DETECTION:
    keypoint_detection_inference()


def keypoint_detection_postprocess():
    global KEYPOINT_DETECTION_CONFIG, DETECTION_NPY_SAVE_PATH
    # (1, 2)
    heatmap_stride_wh = np.array(
        [
            [
                KEYPOINT_DETECTION_CONFIG.data.heatmap_stride[1],
                KEYPOINT_DETECTION_CONFIG.data.heatmap_stride[0],
            ]
        ]
    )
    print("HEATMAP STRIDE WH:", heatmap_stride_wh)

    all_final_kpt = []
    for img_id in tqdm(ALL_IMG_IDS):
        if "VAL" in MODE:
            save_npz_path = os.path.join(
                DETECTION_TMP_DIR, f'{img_id.replace("/", "--")}.npz'
            )
        else:
            save_npz_path = os.path.join(DETECTION_TMP_DIR, f"{img_id}.npz")
        npzfile = np.load(save_npz_path)
        heatmap_grid = npzfile["heatmap_grid"]
        dsnt_main_xy = npzfile["dsnt_main_xy"]
        M = npzfile["M"]
        ori_w, ori_h = npzfile["ori_wh"]

        heatmap_h, heatmap_w = heatmap_grid.shape

        # estimate Homography transformation matrix from DSNT main keypoints
        # H: from current image TO reference image (to_ref_H)
        assert dsnt_main_xy.shape[1:] == (57, 2)
        # (57, 2) -> (NUM_FOLDS, 57, 2)
        ref_main_kpt_xys = REF_MAIN_KPT_XYS[None].repeat(dsnt_main_xy.shape[0], axis=0)
        # ransacReprojThreshold: Even though MAGSAC is "threshold-free", OpenCV still requires
        # this parameter as an upper bound for internal optimizations. 3.0 - 5.0 is standard.
        for ransacReprojThreshold in [3, 5, 7, 9]:
            H, mask = cv2.findHomography(
                dsnt_main_xy.reshape(-1, 2),
                ref_main_kpt_xys.reshape(-1, 2),
                cv2.USAC_MAGSAC,  # MAGSAC++ algorithm
                ransacReprojThreshold=ransacReprojThreshold,
                maxIters=100_000,
                confidence=0.9999,
            )
            if H is not None and mask.sum() / mask.size > 0.1:
                print(
                    f"findHomography mask sum at threshold {ransacReprojThreshold}: {mask.sum()}/{mask.size}"
                )
                break
        else:
            print(
                "FIND HOMOGRAPHY FAILED, FALLBACK TO PERSPECTIVE TRANSFORM:",
                ransacReprojThreshold,
                mask.sum(),
            )
            # ref_kpt_names.index('dc_0_0'), ref_kpt_names.index('s_0_2_t'), ref_kpt_names.index('s_2_2_b'), ref_kpt_names.index('dc_3_3')
            # [2391, 2369, 2386, 2406] - 2365 = [26, 4, 21, 41]
            src_xy = np.median(dsnt_main_xy[:, [26, 4, 21, 41]], axis=0).astype(
                "float32"
            )  # (4, 2)
            dst_xy = REF_MAIN_KPT_XYS[[26, 4, 21, 41]].astype("float32")  # (4, 2)
            H = cv2.getPerspectiveTransform(
                src_xy,
                dst_xy,
            )

        to_ref_H = H
        # estimate relative scale
        from_ref_H = np.linalg.pinv(to_ref_H)
        # roughtly estimate the scale from current original image (000x) relative to the reference image 0001
        ori_scale_x, ori_scale_y = get_scale_xy_from_homo_mat(from_ref_H)

        # radius: ~20 on type1 1700x2200 image
        # image is isotropically resize + padding to `cfg.data.img_size`
        nms_thres_x = 20 * ori_scale_x * min(heatmap_w / ori_w, heatmap_h / ori_h)
        nms_thres_y = 20 * ori_scale_y * min(heatmap_w / ori_w, heatmap_h / ori_h)
        nms_thres = ((nms_thres_x**2 + nms_thres_y**2) / 2) ** 0.5
        nms_thres = max(5, nms_thres)
        # print('NMS THRES:', nms_thres_x, nms_thres_y, nms_thres)

        # decode -> grid points
        # YX order, heatmap pixel indices space
        _t0 = time.time()
        raw_heatmap_grid_kpt_pred = decode_heatmap_2d_batched(
            # step_output["heatmap_all_pred"][:, 0:1],
            torch.from_numpy(heatmap_grid[None, None]).to(DEVICE),  # (1, C, H, W)
            pool_ksize=[3, 3],
            nms_radius_thres=nms_thres,
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
        # we don't handle batch here, just single heatmap at a time
        assert len(raw_heatmap_grid_kpt_pred) == 1
        img_grid_kpt = raw_heatmap_grid_kpt_pred[0]
        assert len(img_grid_kpt) == 1  # 1-channel only
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
            xy = xy * heatmap_stride_wh
            # back to original image space
            xy = batch_perspective_transform_2d(
                xy[None],  # (1, L, 2)
                M[None],  # (1, 3, 3)
            )[0]
            # (x, y, conf) format
            decoded_img_grid_kpt = np.concatenate([xy, img_grid_kpt[:, 2:3]], axis=1)

        # print('AFTER NMS:', decoded_img_grid_kpt.shape)

        # @TODO - ADD TRY EXCEPT HERE
        try:
            final_grid_kpt = register_grid_keypoints(
                decoded_img_grid_kpt,
                to_ref_H,
                viz=False,
                img=None,
                ref_img=None,
                verbose=False,
            )
        except:
            print('UNEXPECTED EXCEPTION IN REGISTERING GRID KEYPOINTS :( ')
            final_grid_kpt = np.zeros((2365, 2), dtype = 'float32')

        final_main_kpt = np.median(dsnt_main_xy, axis=0)
        final_kpt = np.concatenate([final_grid_kpt, final_main_kpt], axis=0)
        assert final_kpt.shape == (2422, 2)
        all_final_kpt.append(final_kpt)

        # remove temporary npz files
        # if DO_REMOVE_TMP_DETECTION:
        #     os.remove(save_npz_path)

    all_final_kpt = np.stack(all_final_kpt, axis=0)
    assert all_final_kpt.shape == (len(ALL_IMG_IDS), 2422, 2)

    # save
    np.save(DETECTION_NPY_SAVE_PATH, all_final_kpt)

    print("DONE DETECTION POSTPROCESSING!!!\n\n\n")

    gc.collect()
    torch.cuda.empty_cache()


if DO_REGISTER:
    keypoint_detection_postprocess()


if DO_VIZ_DETECTION:
    all_detections = np.load(DETECTION_NPY_SAVE_PATH)
    viz_img_idxs = random.choices(
        list(range(len(ALL_IMG_IDS))), k=min(2, len(ALL_IMG_IDS))
    )
    for viz_img_idx in viz_img_idxs:
        img_id = ALL_IMG_IDS[viz_img_idx]
        img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # predicted rotation index, from 0 to 3
        # 0: 0 degree, 1: 90 degree, 2: 180 degree, 3: 270 degree
        rot_code = ROTATION_MAP[img_id]
        # after using this model to predict rotation index
        # standardize the orientation of 0 degree
        std_rotation_code = [
            None,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_CLOCKWISE,
        ][rot_code]
        if std_rotation_code is not None:
            img = cv2.rotate(img, std_rotation_code)
        else:
            # 0 to 0, unchanged
            pass

        detected_kpt_xys = all_detections[viz_img_idx]
        if "KAGGLE" in MODE:
            save_path = None
        else:
            save_path = os.path.join(TMP_DIR, "viz", f"{img_id.replace('/', '--')}.png")
        viz_img_keypoints(
            img,
            kpt_xys=detected_kpt_xys,
            kpt_classes=[0] * 2365 + list(range(57)),
            save_path=save_path,
            figsize=(12, 12),
            point_size=10,
            alpha=0.9,
            show_legend=False,
        )
        print("Done visualization, saved to", save_path)


# ======================== SIGNAL HEATMAP INFERENCE ======================

SIGNAL_RESULT_DIR = os.path.join(TMP_DIR, "signal")
os.makedirs(SIGNAL_RESULT_DIR, exist_ok=True)


def signal_heatmap_inference(cfg, ckpt_path, save_name):
    global ROTATION_MAP
    signal_heatmap_dataset = ECGSignalHeatmapDataset(
        cfg,
        img_ids=ALL_IMG_IDS,
        img_dir=TEST_IMG_DIR,
        template_root_dir=TEMPLATE_ROOT_DIR,
        rotation_map=ROTATION_MAP,
    )
    signal_heatmap_loader = DataLoader(
        signal_heatmap_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=LOADER_NUM_WORKERS,
        drop_last=False,
        pin_memory=False,
    )

    # load model
    signal_heatmap_task = l_utils.build_task(cfg)

    l_utils.load_lightning_state_dict(
        model=signal_heatmap_task,
        ckpt_path=ckpt_path,
        cfg=cfg,
    )
    logger.info("Loaded Pytorch state dict from %s", ckpt_path)
    model = signal_heatmap_task.model
    model.to(DEVICE).eval()
    # print(model)

    # BUILD CODEC
    ds = signal_heatmap_dataset
    heatmap_codec = ColumnGaussianHeatmapCodec(
        sigma=cfg.data.sigma,
        crop_aspect_ratio=ds.CROP_H / ds.CROP_W,
        H=ds.GT_H,
        W=ds.GT_W,
        L=ds.GT_L,
        cutoff_thres=0.011108996538242308,
        adaptive_sigma_scale=cfg.data.adaptive_sigma_scale,
        subpixel=False,
        dtype=torch.float32,
        ref_signal_pixels=constants.REF_SIGNAL_PIXELS
        * cfg.data.signal_length_secs
        / 2.5,
    )

    test_samples = ds.samples

    # PERFORM INFERENCE

    cur_idx = -1
    all_heatmap_preds = {}
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        for batch_idx, batch in tqdm(
            enumerate(signal_heatmap_loader),
            total=len(signal_heatmap_loader),
        ):
            # print(batch_idx, [(k, v.shape) for k, v in batch.items()])
            batch_imgs = batch["image"].to(DEVICE)
            prob_heatmap_pred, _offset_heatmap_pred, _regress_signal_pred = model(
                batch_imgs
            )
            assert _offset_heatmap_pred is None
            # usually nn.Softmax(dim=2)
            prob_heatmap_pred = model.heatmap_act(prob_heatmap_pred)
            # we don't use regress_signal_pred, so don't decode it
            heatmap_signal_pred, _ = heatmap_codec.batch_decode(
                prob_heatmap_pred,
                _offset_heatmap_pred,
                None,  # _regress_signal_pred
            )
            # .float() to deal with bf16
            heatmap_signal_pred = heatmap_signal_pred.cpu().float().numpy()
            B = heatmap_signal_pred.shape[0]
            assert B == 1, "Just to assert, >1 is still okay"

            for i in range(B):
                cur_idx += 1
                img_idx, img_id, crop_name, gt_len, fs = test_samples[cur_idx]

                # interpolate to original length
                ori_heatmap_signal_pred = interp_1d_cv2(
                    heatmap_signal_pred[i], gt_len, method="adaptive"
                )
                all_heatmap_preds.setdefault(img_id, {})[
                    crop_name
                ] = ori_heatmap_signal_pred

    # save predictions result as pickle
    pkl_save_path = os.path.join(SIGNAL_RESULT_DIR, f"{save_name}.pkl")
    os.makedirs(os.path.dirname(pkl_save_path), exist_ok=True)
    with open(pkl_save_path, "wb") as f:
        pickle.dump(all_heatmap_preds, f)
    print("SAVED PREDICTIONS TO", pkl_save_path, "\n\n\n\n\n")


if DO_SIGNAL_HEATMAP_INFERENCE:
    # FOLD 0 COAT 512
    # cfg_path = "./assets/EXP1_COAT512_FOLD0_ep3_step50000_val_SNR23.380785_config.yaml"
    # ckpt_path = "./assets/EXP1_COAT512_FOLD0_ep3_step50000_val_SNR23.380785.ckpt"
    # save_name = "COAT512_FOLD0"
    # cfg = OmegaConf.load(cfg_path)
    # cfg.data.warp.unwrap_flow.kwargs.K = 16
    # cfg.data.use_template = None
    # cfg.data.warp.unwrap_flow._shift_0dot5_in_cv2_remap = True
    # print("MODEL 1 CONFIG:", cfg, sep="\n")
    # signal_heatmap_inference(cfg, ckpt_path, save_name)

    # # FINAL COAT + VGG LR 2e-4
    # cfg_path = "/kaggle/input/ecg-checkpoints/FINAL_01-19__21-15-59.746642_ALLDATA_FINAL_COAT512_VGG1024_GT1000_LR2en4_SEED980611_config.yaml"
    # ckpt_path = "/kaggle/input/ecg-checkpoints/FINAL_01-19__21-15-59.746642_ALLDATA_FINAL_COAT512_VGG1024_GT1000_LR2en4_SEED980611_ep3_step70007_val_SNR28.006985.ckpt"
    # save_name = "ALLDATA_FINAL_COAT512_VGG1024_GT1000_LR2en4"
    # cfg = OmegaConf.load(cfg_path)
    # cfg.model.encoder.pretrained = False
    # cfg.model.fine_encoder.pretrained=False
    # print("MODEL 1 CONFIG:", cfg, sep="\n")
    # signal_heatmap_inference(cfg, ckpt_path, save_name)


    # FINAL COAT + VGG LR 2e-4
    cfg_path = "/kaggle/input/ecg-checkpoints/FINAL_01-19__22-22-04.345114_ALLDATA_FINAL_CONVNEXT512_VGG1024_GT1000_LR1en4_SEED981022_config.yaml"
    ckpt_path = "/kaggle/input/ecg-checkpoints/FINAL_01-19__22-22-04.345114_ALLDATA_FINAL_CONVNEXT512_VGG1024_GT1000_LR1en4_SEED981022_ep3_step60011_val_SNR27.484089.ckpt"
    save_name = "ALLDATA_FINAL_CONVNEXT512_VGG1024_GT1000_LR1en4"
    cfg = OmegaConf.load(cfg_path)
    cfg.model.encoder.pretrained = False
    cfg.model.fine_encoder.pretrained=False
    print("MODEL 1 CONFIG:", cfg, sep="\n")
    signal_heatmap_inference(cfg, ckpt_path, save_name)


# ================== SIGNAL POST PROCESSING =============
# post processing
# merge multiple II_long crop into one
# ensemble (e.g, average) II_short vs II_long

with open(
    os.path.join(SIGNAL_RESULT_DIR, "ALLDATA_FINAL_CONVNEXT512_VGG1024_GT1000_LR1en4.pkl"),
    "rb",
) as f:
    all_heatmap_preds = pickle.load(f)

for img_id, heatmap_preds in all_heatmap_preds.items():
    # print("Postprocessing signal of image", img_id)
    heatmap_preds["II"] = np.concatenate(
        [
            heatmap_preds.pop(k)
            for k in ("II_long_0", "II_long_1", "II_long_2", "II_long_3")
        ],
        axis=0,
    )

    # handle lead II_short
    # simply ignore, or average with II_long_0
    heatmap_preds.pop("II_short")

    # re-interpolate if II_long still not match (happen only when len10 is ood number ?)
    len10 = EXPECTED_GT_LEN[img_id]["II"]
    if len(heatmap_preds["II"]) != len10:
        heatmap_preds["II"] = interp_1d_cv2(heatmap_preds["II"], len10)


final_preds = all_heatmap_preds
assert len(heatmap_preds) == 12

# for img_id, img_preds in final_preds.items():
#     print(
#         img_id,
#         "-->",
#         {lead_name: len(signal) for lead_name, signal in img_preds.items()},
#     )

# print(final_preds)


if MODE != "KAGGLE_TEST" and DO_EVALUATE:
    # grab the GT
    all_gts = {}
    for img_id in ALL_IMG_IDS:
        all_gts[img_id] = load_sample_signal(TEST_IMG_DIR, img_id)

    from ecg.utils.comp_metrics_fast import compute_metrics

    global_snr, per_sample_snrs = compute_metrics(
        final_preds,
        all_gts,
        EXPECTED_GT_FS,
        do_align=True,
    )

    print("\n\n\n================ EVALUATION ==================")
    # print('PER SAMPLE SNR:', per_sample_snrs, sep = '\n')
    print("GLOBAL SNR:", global_snr)


# ==================== FINAL STEP: SUBMISSION PARQUET ==================

if MODE == 'KAGGLE_TEST':
    clear_dir_content('/kaggle/working/')

create_submission_file(
    predictions_dict=final_preds,
    test_csv_path=TEST_CSV_PATH,
    output_path=SUBMISSION_CSV_PATH,
)
print("DONE ! SUBMISSION PARQUET SAVED !")


if 1:
    print("=========== TEST RECOVERING ========")
    submission_df = pd.read_csv(SUBMISSION_CSV_PATH)
    print("SUBMISSION DF:", submission_df.shape)
    # print(submission_df)
    display(submission_df)

if 0:
    recovered_preds = csv_to_nested_dict(SUBMISSION_CSV_PATH)
    global_snr, per_sample_snrs = compute_metrics(
        recovered_preds,
        all_gts,
        EXPECTED_GT_FS,
        do_align=True,
    )
    print("GLOBAL SNR:", global_snr)


print("ALL DONE!")
