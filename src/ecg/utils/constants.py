import gc
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


NUM_GRID_ROWS, NUM_GRID_COLS = 43, 55
NUM_GRID_POINTS = NUM_GRID_ROWS * NUM_GRID_COLS
NUM_MAIN_POINTS = 57
NUM_KPTS = 2422
assert NUM_MAIN_POINTS + NUM_GRID_POINTS == NUM_KPTS
KPT_CLASSES = [0] * NUM_GRID_POINTS + list(range(1, NUM_MAIN_POINTS + 1))
KPT_CLASSES_ARR = np.array(KPT_CLASSES)[..., None]  # (N, 1)

# REFERENCE PLOT/FIGURE DETAILS
REF_IMG_H = 1700
REF_IMG_W = 2200
REF_PLOT_MAX_Y = 21.59
REF_PLOT_MAX_X = 11.176
REF_Y_UNIT_PIXELS = REF_IMG_H / REF_PLOT_MAX_Y  # number of pixels per Y unit
REF_SIGNAL_PIXELS = 492.12598425196853  # number of pixels covered by 2.5 sec signal


# Reference keypoints
cur_dir = os.path.dirname(os.path.abspath(__file__))
ref_kpt_path = os.path.join(cur_dir, "reference_keypoints.json")
logger.info("Loading reference keypoints from %s", ref_kpt_path)
with open(ref_kpt_path, "r") as f:
    ref_kpts_metadata = json.load(f)
REF_KPT_NAMES = list(ref_kpts_metadata.keys())
assert len(REF_KPT_NAMES) == NUM_KPTS
REF_KPT_XYS = np.array(list(ref_kpts_metadata.values()), dtype="float32")
assert REF_KPT_XYS.shape == (NUM_KPTS, 2)
del ref_kpts_metadata
gc.collect()

REF_GRID_KPT_INDICES = np.arange(0, 43 * 55).reshape(43, 55)
REF_FLAT_GRID_KPT_INDICES = REF_GRID_KPT_INDICES.flatten()


ALL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_TO_SIG_LEN = {
    "I": (0, 1 / 4),
    "II": (0, 1.0),
    "III": (0, 1 / 4),
    "aVR": (1 / 4, 2 / 4),
    "aVL": (1 / 4, 2 / 4),
    "aVF": (1 / 4, 2 / 4),
    "V1": (2 / 4, 3 / 4),
    "V2": (2 / 4, 3 / 4),
    "V3": (2 / 4, 3 / 4),
    "V4": (3 / 4, 1),
    "V5": (3 / 4, 1),
    "V6": (3 / 4, 1),
}

# mapping lead name to (left_x, mid_y) of each lead region/bbox
LEAD_LEFT_MID_KEYPOINTS = {
    # row 0
    "I": [118.11023622047246, 708.3333333333335],
    "aVR": [610.236220472441, 708.3333333333335],
    "V1": [1102.3622047244096, 708.3333333333335],
    "V4": [1594.488188976378, 708.3333333333335],
    # row 1
    "II_short": [118.11023622047246, 991.6666666666665],
    "aVL": [610.236220472441, 991.6666666666665],
    "V2": [1102.3622047244096, 991.6666666666665],
    "V5": [1594.488188976378, 991.6666666666665],
    # row 2
    "III": [118.11023622047246, 1275.0],
    "aVF": [610.236220472441, 1275.0],
    "V3": [1102.3622047244096, 1275.0],
    "V6": [1594.488188976378, 1275.0],
    # row 3
    "II_long": [118.11023622047246, 1534.711286089239],
    "II_long_0": [118.11023622047246, 1534.711286089239],
    "II_long_1": [610.236220472441, 1534.711286089239],
    "II_long_2": [1102.3622047244096, 1534.711286089239],
    "II_long_3": [1594.488188976378, 1534.711286089239],
}


# mapping lead name to "nearest" or "relevent" main keypoints
MANUAL_LEAD_TO_NEARBY_MAIN_KEYPOINT_NAMES = {
    "I": ["dc_0_0", "dc_0_1", "dc_0_2", "dc_0_3", "s_0_0_t", "s_0_0_b", "n_I", "n_aVR"],
    "aVR": ["s_0_0_t", "s_0_0_b", "s_0_1_t", "s_0_1_b", "n_aVR", "n_V1"],
    "V1": ["s_0_1_t", "s_0_1_b", "s_0_2_t", "s_0_2_b", "n_V1", "n_V4"],
    "V4": ["s_0_2_t", "s_0_2_b", "s_0_3_t", "s_0_3_b", "n_V4"],
    "II_short": [
        "dc_1_0",
        "dc_1_1",
        "dc_1_2",
        "dc_1_3",
        "s_1_0_t",
        "s_1_0_b",
        "n_II",
        "n_aVL",
    ],
    "aVL": ["s_1_0_t", "s_1_0_b", "s_1_1_t", "s_1_1_b", "n_aVL", "n_V2"],
    "V2": ["s_1_1_t", "s_1_1_b", "s_1_2_t", "s_1_2_b", "n_V2", "n_V5"],
    "V5": ["s_1_2_t", "s_1_2_b", "s_1_3_t", "s_1_3_b", "n_V5"],
    "III": [
        "dc_2_0",
        "dc_2_1",
        "dc_2_2",
        "dc_2_3",
        "s_2_0_t",
        "s_2_0_b",
        "n_III",
        "n_aVF",
    ],
    "aVF": ["s_2_0_t", "s_2_0_b", "s_2_1_t", "s_2_1_b", "n_aVF", "n_V3"],
    "V3": ["s_2_1_t", "s_2_1_b", "s_2_2_t", "s_2_2_b", "n_V3", "n_V6"],
    "V6": ["s_2_2_t", "s_2_2_b", "s_2_3_t", "s_2_3_b", "n_V6"],
    "II_long": [
        "dc_3_0",
        "dc_3_1",
        "dc_3_2",
        "dc_3_3",
        "s_3_3_t",
        "s_3_3_b",
        "n_II_full",
        "mms",
        "mmmv",
    ],
    "II_long_0": [
        "dc_3_0",
        "dc_3_1",
        "dc_3_2",
        "dc_3_3",
        "s_3_3_t",
        "s_3_3_b",
        "n_II_full",
        "mms",
        "mmmv",
    ],
    "II_long_1": [
        "dc_3_0",
        "dc_3_1",
        "dc_3_2",
        "dc_3_3",
        "s_3_3_t",
        "s_3_3_b",
        "n_II_full",
        "mms",
        "mmmv",
    ],
    "II_long_2": [
        "dc_3_0",
        "dc_3_1",
        "dc_3_2",
        "dc_3_3",
        "s_3_3_t",
        "s_3_3_b",
        "n_II_full",
        "mms",
        "mmmv",
    ],
    "II_long_3": [
        "dc_3_0",
        "dc_3_1",
        "dc_3_2",
        "dc_3_3",
        "s_3_3_t",
        "s_3_3_b",
        "n_II_full",
        "mms",
        "mmmv",
    ],
}
MANUAL_LEAD_TO_NEARBY_MAIN_KEYPOINT_INDICES = {
    lead_name: [REF_KPT_NAMES.index(kpt_name) for kpt_name in nearby_kpt_names]
    for lead_name, nearby_kpt_names in MANUAL_LEAD_TO_NEARBY_MAIN_KEYPOINT_NAMES.items()
}


AUTO_LEAD_TO_NEARBY_MAIN_KEYPOINT_NAMES = {'I': ['n_I', 'dc_0_2', 's_0_0_t', 's_0_0_b', 'dc_0_1', 'n_aVR', 'dc_0_3', 'dc_0_0', 'dc_1_1', 'dc_1_0', 's_1_0_t', 'dc_1_2', 's_1_0_b', 'dc_1_3', 'n_II', 'n_aVL', 'dc_2_1', 'dc_2_0', 's_2_0_t', 'dc_2_2', 'dc_2_3', 's_2_0_b', 'n_III', 'n_aVF', 's_0_1_t', 's_0_1_b', 'n_V1', 's_1_1_t', 'dc_3_1', 's_1_1_b', 'dc_3_0', 'n_V2', 'dc_3_2', 'dc_3_3', 'n_II_full', 's_2_1_t', 's_2_1_b', 'mms', 'n_V3', 'mmmv', 's_0_2_t', 's_0_2_b', 'n_V4', 's_1_2_t', 's_1_2_b', 'n_V5', 's_2_2_t', 's_2_2_b', 'n_V6', 's_0_3_t', 's_0_3_b', 's_1_3_t', 's_1_3_b', 's_2_3_t', 's_2_3_b', 's_3_3_t', 's_3_3_b'], 'aVR': ['n_aVR', 's_0_0_t', 's_0_0_b', 's_0_1_t', 's_0_1_b', 'n_V1', 's_1_0_t', 's_1_1_t', 's_1_0_b', 's_1_1_b', 'n_aVL', 'n_V2', 's_2_0_t', 's_2_1_t', 's_2_0_b', 's_2_1_b', 'n_aVF', 'n_V3', 'n_I', 'dc_0_2', 's_0_2_t', 's_0_2_b', 'dc_0_1', 'n_V4', 'dc_1_1', 'dc_0_3', 'dc_0_0', 's_1_2_t', 'dc_1_2', 'n_II', 's_1_2_b', 'dc_1_0', 'dc_1_3', 'n_V5', 'dc_2_1', 's_2_2_t', 'dc_2_0', 'dc_2_2', 's_2_2_b', 'n_III', 'mmmv', 'dc_2_3', 'n_V6', 'dc_3_1', 'mms', 'dc_3_0', 'dc_3_2', 'dc_3_3', 'n_II_full', 's_0_3_t', 's_0_3_b', 's_1_3_t', 's_1_3_b', 's_2_3_t', 's_2_3_b', 's_3_3_t', 's_3_3_b'], 'V1': ['n_V1', 's_0_1_t', 's_0_1_b', 's_0_2_t', 's_0_2_b', 'n_V4', 's_1_1_t', 's_1_2_t', 's_1_1_b', 's_1_2_b', 'n_V2', 'n_V5', 's_2_1_t', 's_2_2_t', 's_2_1_b', 's_2_2_b', 'n_V3', 'n_V6', 'n_aVR', 's_0_3_t', 's_0_3_b', 's_0_0_t', 's_0_0_b', 's_1_3_t', 'n_aVL', 's_1_0_t', 's_1_3_b', 's_1_0_b', 's_2_3_t', 's_2_0_t', 'n_aVF', 's_2_3_b', 's_2_0_b', 's_3_3_t', 'mmmv', 's_3_3_b', 'n_I', 'dc_0_2', 'dc_0_1', 'dc_1_1', 'dc_1_2', 'n_II', 'dc_0_3', 'dc_0_0', 'dc_1_0', 'dc_1_3', 'dc_2_1', 'mms', 'dc_2_2', 'dc_2_0', 'n_III', 'dc_2_3', 'dc_3_1', 'dc_3_0', 'dc_3_2', 'n_II_full', 'dc_3_3'], 'V4': ['n_V4', 's_0_3_t', 's_0_3_b', 's_0_2_t', 's_0_2_b', 's_1_3_t', 's_1_2_t', 's_1_3_b', 's_1_2_b', 'n_V5', 's_2_3_t', 's_2_2_t', 's_2_3_b', 's_2_2_b', 'n_V6', 'n_V1', 's_0_1_t', 's_0_1_b', 's_1_1_t', 'n_V2', 's_1_1_b', 's_3_3_t', 's_3_3_b', 's_2_1_t', 'n_V3', 's_2_1_b', 'n_aVR', 's_0_0_t', 's_0_0_b', 'n_aVL', 's_1_0_t', 's_1_0_b', 's_2_0_t', 'n_aVF', 's_2_0_b', 'mmmv', 'n_I', 'dc_0_2', 'dc_0_1', 'mms', 'dc_1_1', 'n_II', 'dc_1_2', 'dc_0_3', 'dc_0_0', 'dc_1_0', 'dc_1_3', 'dc_2_1', 'dc_2_2', 'n_III', 'dc_2_0', 'dc_2_3', 'dc_3_1', 'dc_3_2', 'dc_3_0', 'n_II_full', 'dc_3_3'], 'II_short': ['n_II', 'dc_1_2', 's_1_0_b', 's_1_0_t', 'dc_1_1', 'n_aVL', 'dc_1_3', 'dc_1_0', 'dc_2_1', 'n_I', 'dc_2_0', 's_2_0_t', 's_0_0_b', 'n_aVR', 'dc_2_2', 'dc_0_2', 's_2_0_b', 's_0_0_t', 'dc_2_3', 'dc_0_3', 'n_III', 'n_aVF', 'dc_0_1', 'dc_0_0', 'dc_3_1', 'dc_3_0', 'dc_3_2', 'dc_3_3', 'n_II_full', 'mms', 's_1_1_b', 's_1_1_t', 'n_V2', 's_2_1_t', 's_0_1_b', 'mmmv', 'n_V1', 's_2_1_b', 's_0_1_t', 'n_V3', 's_1_2_b', 's_1_2_t', 'n_V5', 's_2_2_t', 's_0_2_b', 's_2_2_b', 's_0_2_t', 'n_V4', 'n_V6', 's_1_3_b', 's_1_3_t', 's_2_3_t', 's_0_3_b', 's_2_3_b', 's_0_3_t', 's_3_3_t', 's_3_3_b'], 'aVL': ['n_aVL', 's_1_0_b', 's_1_0_t', 's_1_1_b', 's_1_1_t', 'n_V2', 'n_aVR', 's_2_0_t', 's_2_1_t', 's_0_0_b', 's_0_1_b', 'n_V1', 's_2_0_b', 's_2_1_b', 's_0_0_t', 's_0_1_t', 'n_aVF', 'n_V3', 'mmmv', 'n_II', 'dc_1_2', 's_1_2_b', 's_1_2_t', 'dc_1_1', 'n_V5', 'dc_2_1', 'n_I', 'dc_1_3', 'dc_1_0', 's_2_2_t', 's_0_2_b', 'dc_2_2', 'dc_0_2', 'n_V4', 'n_III', 's_2_2_b', 's_0_2_t', 'dc_2_0', 'mms', 'dc_0_1', 'dc_2_3', 'dc_0_3', 'n_V6', 'dc_0_0', 'dc_3_1', 'dc_3_0', 'dc_3_2', 'n_II_full', 'dc_3_3', 's_1_3_b', 's_1_3_t', 's_2_3_t', 's_0_3_b', 's_2_3_b', 's_0_3_t', 's_3_3_t', 's_3_3_b'], 'V2': ['n_V2', 's_1_1_b', 's_1_1_t', 's_1_2_b', 's_1_2_t', 'n_V5', 'n_V1', 's_2_1_t', 's_0_1_b', 's_2_2_t', 's_0_2_b', 'n_V4', 's_2_1_b', 's_0_1_t', 's_2_2_b', 's_0_2_t', 'n_V3', 'n_V6', 'n_aVL', 's_1_3_b', 's_1_3_t', 's_1_0_b', 's_1_0_t', 'n_aVR', 's_2_3_t', 's_0_3_b', 'n_aVF', 's_2_0_t', 's_0_0_b', 's_2_3_b', 's_0_3_t', 's_2_0_b', 's_0_0_t', 'mmmv', 's_3_3_t', 's_3_3_b', 'mms', 'n_II', 'dc_1_2', 'dc_1_1', 'dc_2_1', 'n_I', 'n_III', 'dc_2_2', 'dc_0_2', 'dc_1_3', 'dc_1_0', 'dc_0_1', 'dc_2_0', 'dc_2_3', 'dc_0_3', 'dc_3_1', 'dc_0_0', 'dc_3_2', 'dc_3_0', 'n_II_full', 'dc_3_3'], 'V5': ['n_V5', 's_1_3_b', 's_1_3_t', 's_1_2_b', 's_1_2_t', 'n_V4', 's_2_3_t', 's_0_3_b', 's_2_2_t', 's_0_2_b', 's_2_3_b', 's_0_3_t', 's_2_2_b', 's_0_2_t', 'n_V6', 's_3_3_t', 's_3_3_b', 'n_V2', 's_1_1_b', 's_1_1_t', 'n_V1', 's_2_1_t', 's_0_1_b', 'n_V3', 's_2_1_b', 's_0_1_t', 'n_aVL', 'n_aVR', 's_1_0_b', 's_1_0_t', 'n_aVF', 'mmmv', 's_2_0_t', 's_0_0_b', 's_2_0_b', 's_0_0_t', 'mms', 'n_II', 'dc_1_2', 'dc_1_1', 'n_I', 'dc_2_1', 'n_III', 'dc_2_2', 'dc_0_2', 'dc_0_1', 'dc_1_3', 'dc_1_0', 'dc_2_0', 'dc_3_1', 'dc_2_3', 'dc_0_3', 'dc_0_0', 'dc_3_2', 'n_II_full', 'dc_3_0', 'dc_3_3'], 'III': ['n_III', 'dc_2_2', 's_2_0_b', 's_2_0_t', 'dc_2_1', 'n_aVF', 'dc_2_3', 'dc_2_0', 'dc_3_1', 'n_II', 'dc_3_0', 's_1_0_b', 'dc_3_2', 'n_aVL', 'dc_1_2', 'dc_3_3', 'mms', 'n_II_full', 's_1_0_t', 'dc_1_3', 'dc_1_1', 'dc_1_0', 'n_I', 'mmmv', 'n_aVR', 's_0_0_b', 'dc_0_2', 'dc_0_3', 's_0_0_t', 'dc_0_1', 'dc_0_0', 's_2_1_t', 's_2_1_b', 'n_V3', 's_1_1_b', 'n_V2', 's_1_1_t', 's_0_1_b', 'n_V1', 's_0_1_t', 's_2_2_t', 's_2_2_b', 'n_V6', 's_1_2_b', 's_1_2_t', 'n_V5', 's_0_2_b', 'n_V4', 's_0_2_t', 's_2_3_b', 's_2_3_t', 's_3_3_t', 's_1_3_b', 's_3_3_b', 's_1_3_t', 's_0_3_b', 's_0_3_t'], 'aVF': ['n_aVF', 's_2_0_b', 's_2_0_t', 's_2_1_b', 's_2_1_t', 'n_V3', 'n_aVL', 's_1_0_b', 's_1_1_b', 'n_V2', 'mmmv', 's_1_0_t', 's_1_1_t', 'n_aVR', 'n_V1', 's_0_0_b', 's_0_1_b', 'mms', 's_0_0_t', 's_0_1_t', 'n_III', 'dc_2_2', 's_2_2_b', 's_2_2_t', 'dc_2_1', 'dc_3_1', 'n_V6', 'n_II', 'dc_2_3', 'dc_2_0', 'dc_3_2', 's_1_2_b', 'dc_1_2', 'n_II_full', 'n_V5', 'dc_3_0', 's_1_2_t', 'dc_3_3', 'dc_1_1', 'dc_1_3', 'dc_1_0', 'n_I', 's_0_2_b', 'n_V4', 'dc_0_2', 's_0_2_t', 'dc_0_3', 'dc_0_1', 'dc_0_0', 's_2_3_b', 's_2_3_t', 's_3_3_t', 's_1_3_b', 's_3_3_b', 's_1_3_t', 's_0_3_b', 's_0_3_t'], 'V3': ['n_V3', 's_2_1_t', 's_2_1_b', 's_2_2_t', 's_2_2_b', 'n_V6', 'n_V2', 's_1_1_b', 's_1_2_b', 'n_V5', 's_1_1_t', 's_1_2_t', 'n_V1', 'n_V4', 's_0_1_b', 's_0_2_b', 's_0_1_t', 's_0_2_t', 'mmmv', 'n_aVF', 's_2_3_b', 's_2_3_t', 's_2_0_b', 's_2_0_t', 'n_aVL', 's_3_3_t', 's_1_3_b', 's_1_0_b', 's_3_3_b', 's_1_3_t', 's_1_0_t', 'n_aVR', 's_0_3_b', 's_0_0_b', 's_0_3_t', 's_0_0_t', 'mms', 'n_III', 'dc_2_2', 'dc_2_1', 'dc_3_1', 'n_II', 'dc_3_2', 'n_II_full', 'dc_1_2', 'dc_2_3', 'dc_2_0', 'dc_1_1', 'dc_3_0', 'dc_3_3', 'dc_1_3', 'dc_1_0', 'n_I', 'dc_0_2', 'dc_0_1', 'dc_0_3', 'dc_0_0'], 'V6': ['n_V6', 's_2_3_b', 's_2_3_t', 's_2_2_b', 's_2_2_t', 'n_V5', 's_3_3_t', 's_1_3_b', 's_1_2_b', 's_3_3_b', 's_1_3_t', 's_1_2_t', 'n_V4', 's_0_3_b', 's_0_2_b', 's_0_3_t', 's_0_2_t', 'n_V3', 's_2_1_b', 's_2_1_t', 'n_V2', 's_1_1_b', 's_1_1_t', 'n_V1', 's_0_1_b', 's_0_1_t', 'mmmv', 'n_aVF', 'n_aVL', 's_2_0_b', 's_2_0_t', 's_1_0_b', 's_1_0_t', 'n_aVR', 's_0_0_b', 's_0_0_t', 'mms', 'n_III', 'dc_2_2', 'dc_2_1', 'n_II', 'dc_3_1', 'dc_3_2', 'n_II_full', 'dc_1_2', 'dc_1_1', 'dc_2_3', 'dc_2_0', 'dc_3_0', 'dc_3_3', 'dc_1_3', 'n_I', 'dc_1_0', 'dc_0_2', 'dc_0_1', 'dc_0_3', 'dc_0_0'], 'II_long': ['n_V3', 's_2_1_b', 's_2_1_t', 'mmmv', 'n_V2', 'n_aVF', 's_1_1_b', 's_2_0_b', 's_2_2_b', 'n_V6', 's_1_1_t', 's_2_0_t', 's_2_2_t', 'n_aVL', 'n_V5', 's_1_0_b', 's_1_2_b', 'mms', 's_1_0_t', 's_1_2_t', 'n_V1', 's_0_1_b', 's_0_1_t', 'n_aVR', 'n_V4', 's_0_0_b', 's_0_2_b', 'n_II_full', 's_0_0_t', 's_0_2_t', 'dc_3_2', 's_3_3_b', 's_3_3_t', 'dc_3_1', 'n_III', 's_2_3_b', 'dc_2_2', 'dc_3_3', 's_2_3_t', 'dc_3_0', 'dc_2_1', 'dc_2_3', 'dc_2_0', 'n_II', 's_1_3_b', 'dc_1_2', 's_1_3_t', 'dc_1_3', 'dc_1_1', 'dc_1_0', 'n_I', 's_0_3_b', 'dc_0_2', 's_0_3_t', 'dc_0_3', 'dc_0_1', 'dc_0_0']}  # fmt: skip
