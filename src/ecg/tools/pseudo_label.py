import json
import os
import pickle
from typing import Tuple

import cv2
import numpy as np
import polars as pl
import scipy.optimize
import scipy.spatial
from numba import njit
from PIL import Image
from tqdm import tqdm
from yagm.utils.geometric import batch_perspective_transform_2d

from ecg.utils.misc import get_scale_xy_from_homo_mat
from ecg.utils.viz import viz_img_keypoints


def check_points_inside_polygon(keypoints, polygon_coords):
    """
    Generates a boolean mask for keypoints lying inside a polygon using OpenCV.

    Args:
        keypoints (np.ndarray): Shape (L, 2) in (x, y) order. Can be float or int.
        polygon_coords (np.ndarray): Shape (4, 2) in (x, y) order.
                                     Defined as [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    Returns:
        np.ndarray: A boolean mask of shape (L,) where True indicates the point
                    is inside or on the edge of the polygon.
    """
    # 1. OpenCV contours require int32 (for pixel precision) or float32.
    # We cast to int32 here as per your provided snippet, which is standard for image polygons.
    poly_array = np.array(polygon_coords, dtype=np.int32)

    mask = []

    # 2. Iterate through the (L, 2) array
    for point in keypoints:
        # pointPolygonTest requires the point as a tuple (x, y).
        # We allow float coordinates for the points even if the polygon is int.
        pt_tuple = (float(point[0]), float(point[1]))

        # measureDist=False returns:
        # +1 if inside, -1 if outside, 0 if on the edge.
        result = cv2.pointPolygonTest(poly_array, pt_tuple, measureDist=False)

        # We consider "inside" or "on edge" (>= 0) as True
        mask.append(result >= 0)

    return np.array(mask, dtype=bool)


@njit
def greedy_match(pred_xy, pred_score, gt_xy, threshold, cost_gating=None):
    """
    Greedy confidence-ordered matching for ONE image.

    Returns:
        matches (np.ndarray): Shape (K, 2) [Pred Index, GT Index]
        unmatched_preds (np.ndarray): Shape (P-K, ) Indices of unmatched predictions
        unmatched_gts (np.ndarray): Shape (G-K, ) Indices of unmatched ground truths
    """
    del cost_gating
    P = pred_xy.shape[0]
    G = gt_xy.shape[0]

    # 1. Sort predictions by score (descending)
    order = np.argsort(-pred_score)
    sorted_pred_xy = pred_xy[order]

    # 2. Initialize tracking
    gt_used = np.zeros(G, dtype=np.bool_)
    pred_used_sorted_idx = np.zeros(P, dtype=np.bool_)  # Tracks usage in sorted order

    # Pre-allocate matches array (max matches = min(P, G))
    max_matches = min(P, G)
    temp_matches = np.zeros((max_matches, 2), dtype=np.int64)
    match_count = 0

    # 3. Greedy Matching Loop
    for i in range(P):
        if G == 0:
            break

        min_dist = 1e12
        min_j = -1

        # Find closest unused GT
        for j in range(G):
            if gt_used[j]:
                continue

            dx = gt_xy[j, 0] - sorted_pred_xy[i, 0]
            dy = gt_xy[j, 1] - sorted_pred_xy[i, 1]
            d = (dx * dx + dy * dy) ** 0.5

            if d < min_dist:
                min_dist = d
                min_j = j

        # Check threshold
        if min_j >= 0 and min_dist <= threshold:
            gt_used[min_j] = True
            pred_used_sorted_idx[i] = True

            # Store match: (Original Pred Index, GT Index)
            temp_matches[match_count, 0] = order[i]
            temp_matches[match_count, 1] = min_j
            match_count += 1

    # 4. Finalize Matches
    matches = temp_matches[:match_count]

    # 5. Determine Unmatched Indices
    # Unmatched GTs: straightforward, just check the boolean mask
    # Numba supports boolean indexing like NumPy
    all_gt_indices = np.arange(G)
    unmatched_gts = all_gt_indices[~gt_used]

    # Unmatched Preds: need to map back to original indices
    # We kept track of 'pred_used' in the *sorted* order.
    # We need to find which indices in 'order' were NOT marked as used.

    # Create a boolean mask for original indices
    pred_used_original = np.zeros(P, dtype=np.bool_)
    for i in range(P):
        if pred_used_sorted_idx[i]:
            original_idx = order[i]
            pred_used_original[original_idx] = True

    all_pred_indices = np.arange(P)
    unmatched_preds = all_pred_indices[~pred_used_original]

    return matches, unmatched_preds, unmatched_gts


# Small epsilon to handle floating point comparisons or gating logic
EPSILON = 1e-5
POS_INF = np.finfo(np.float32).max


def _linear_assignment_hungarian(
    cost_matrix: np.ndarray, max_cost: float, cost_gating=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear assignment using Scipy's Hungarian method implementation.
    Provided implementation adapted for the wrapper.
    """
    # Create a copy to avoid modifying the original cost matrix input
    cost_matrix = cost_matrix.copy()
    # print('MIN COST:', cost_matrix.min(), max_cost)

    if cost_gating is not None:
        # Gate costs: treated every over-range cost equally to remove bias
        # cost_matrix[cost_matrix > cost_gating] = cost_gating + EPSILON
        cost_matrix[cost_matrix > cost_gating] = 999999

    # Perform the Hungarian Algorithm (Munkres algorithm)
    # Returns (row_indices, col_indices)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

    matches = []
    for ia, ib in zip(row_ind, col_ind):
        # Filter out matches that exceed the maximum allowed threshold
        # if 1:
        if cost_matrix[ia, ib] <= max_cost:
            matches.append([ia, ib])

    matches = np.asarray(matches)

    # Determine unmatched indices
    if len(matches) > 0:
        unmatched_a = np.asarray(
            [idx for idx in range(cost_matrix.shape[0]) if idx not in matches[:, 0]]
        )
        unmatched_b = np.asarray(
            [idx for idx in range(cost_matrix.shape[1]) if idx not in matches[:, 1]]
        )
    else:
        # If no matches, everything is unmatched
        unmatched_a = np.arange(cost_matrix.shape[0])
        unmatched_b = np.arange(cost_matrix.shape[1])
        # Ensure matches is an empty array of shape (0, 2) for consistency
        matches = np.empty((0, 2), dtype=int)

    return matches, unmatched_a, unmatched_b


def hungarian_match(pred_xy, pred_score, gt_xy, threshold, cost_gating=None):
    """
    Optimal (Hungarian) matching for ONE image.

    Args:
        pred_xy (np.ndarray): Shape (P, 2)
        pred_score (np.ndarray): Shape (P,) - Not used for matching logic in Hungarian,
                                 kept for interface consistency.
        gt_xy (np.ndarray): Shape (G, 2)
        threshold (float): Maximum euclidean distance allowed for a match.

    Returns:
        matches (np.ndarray): Shape (K, 2) [Pred Index, GT Index]
        unmatched_preds (np.ndarray): Shape (P-K, ) Indices of unmatched predictions
        unmatched_gts (np.ndarray): Shape (G-K, ) Indices of unmatched ground truths
    """
    P = pred_xy.shape[0]
    G = gt_xy.shape[0]

    # Handle Edge Cases: Empty inputs
    if P == 0:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=int), np.arange(G)
    if G == 0:
        return np.empty((0, 2), dtype=int), np.arange(P), np.empty(0, dtype=int)

    # 1. Compute Cost Matrix (Euclidean Distance)
    # Shape will be (P, G)
    cost_matrix = scipy.spatial.distance.cdist(pred_xy, gt_xy, metric="euclidean")

    # 2. Perform Hungarian Matching
    matches, unmatched_preds, unmatched_gts = _linear_assignment_hungarian(
        cost_matrix, threshold, cost_gating=cost_gating
    )

    return matches, unmatched_preds, unmatched_gts


def visualize_matches(
    image_rgb, matches, unmatched_preds, unmatched_gts, pred_xy, gt_xy
):
    """
    Visualizes matching results on the provided image.

    Args:
        image_rgb (np.ndarray): Input image (H, W, 3).
        matches (np.ndarray): Shape (K, 2) [Pred Index, GT Index].
        unmatched_preds (np.ndarray): Indices of unmatched predictions.
        unmatched_gts (np.ndarray): Indices of unmatched ground truths.
        pred_xy (np.ndarray): All prediction coordinates (P, 2).
        gt_xy (np.ndarray): All ground truth coordinates (G, 2).

    Returns:
        np.ndarray: The annotated image.
    """
    # 1. Setup Canvas
    # OpenCV uses BGR by default, so if input is RGB, convert it or just treat channels accordingly.
    # We will assume we want to return a BGR image for cv2.imwrite/imshow usage.
    # If image_rgb is None, create a black canvas 1700x2300 as requested.
    if image_rgb is None:
        canvas = np.zeros((1700, 2200, 3), dtype=np.uint8)
    else:
        # Convert RGB to BGR for OpenCV drawing functions
        canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # 2. Define Colors (BGR format)
    COLOR_MATCH = (0, 255, 0)  # Green
    COLOR_UNMATCH_PRED = (0, 0, 255)  # Red (False Positive)
    COLOR_UNMATCH_GT = (255, 0, 0)  # Blue (False Negative / Missed)

    # 3. Draw Unmatched Ground Truths (Blue Circles)
    if gt_xy is not None:
        for idx in unmatched_gts:
            pt = tuple(gt_xy[idx].astype(int))
            # Draw a hollow circle to represent GT
            cv2.circle(canvas, pt, radius=8, color=COLOR_UNMATCH_GT, thickness=2)

    # 4. Draw Unmatched Predictions (Red Crosses/Dots)
    for idx in unmatched_preds:
        pt = tuple(pred_xy[idx].astype(int))
        # Draw a filled circle to represent Prediction
        cv2.circle(canvas, pt, radius=5, color=COLOR_UNMATCH_PRED, thickness=-1)

    # 5. Draw Matches: Blue Lines connecting Pred (RED DOT) to GT (GREEN CIRCLE)
    for pred_idx, gt_idx in matches:
        pt_pred = tuple(pred_xy[pred_idx].astype(int))
        if gt_xy is not None:
            pt_gt = tuple(gt_xy[gt_idx].astype(int))
            # Draw Line
            # cv2.line(canvas, pt_pred, pt_gt, COLOR_MATCH, thickness=2)
            cv2.line(canvas, pt_pred, pt_gt, (0, 0, 255), thickness=2)
            # GT as larger hollow circle
            cv2.circle(canvas, pt_gt, radius=8, color=COLOR_MATCH, thickness=2)

        # Draw endpoints to see the shift
        # Pred as small filled dot
        cv2.circle(canvas, pt_pred, radius=4, color=(255, 0, 0), thickness=-1)

    return canvas


def load_img(sample):
    sample_id = sample["id"]
    type_id = sample["type_id"]
    rot_code = sample["rot_code"]

    img_path = os.path.join(
        "/home/dangnh36/datasets/ecg/raw/train/",
        str(sample_id),
        f"{sample_id}-{type_id:04d}.png",
    )
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # correct the orientation
    if rot_code is not None:
        assert rot_code == int(rot_code)
        rot_code = int(rot_code)
        img = cv2.rotate(img, rot_code)
    return img


def filter_matches_largest_cca(
    matched_ori_gt_idxs, matched_idxs, unmatched_pred_idxs, unmatched_gt_idxs
):
    # 1. Create Grid Mask
    matched_mask = np.zeros((NUM_GRID_KPT,), dtype="uint8")
    matched_mask[matched_ori_gt_idxs] = 1
    matched_mask = matched_mask.reshape(43, 55)

    # 2. Find Connected Components using OpenCV
    # connectivity=8 allows diagonal connections, which is important for grid continuity
    num_labels, labels = cv2.connectedComponents(matched_mask, connectivity=4)

    # 3. Calculate size of each component
    seg_counts = np.bincount(labels.flatten())

    # 4. Find the largest component, ignoring the background (label 0)
    if len(seg_counts) > 1:
        seg_counts[0] = -999  # Zero out background so it's not selected
        largest_label = np.argmax(seg_counts)

        # 5. Create filter mask
        # Flatten labels back to 1D (2365,) to verify matches directly
        labels_flat = labels.flatten()

        # Check if the GT index (col 1) of each match belongs to the largest label
        is_in_largest_cc = labels_flat[matched_ori_gt_idxs] == largest_label

        # Apply split
        prune_matched_idxs = matched_idxs[~is_in_largest_cc]
        matched_idxs = matched_idxs[is_in_largest_cc]
    else:
        # Edge case: No foreground pixels found
        prune_matched_idxs = matched_idxs
        matched_idxs = np.empty((0, 2), dtype=int)

    # 6. Recycle Pruned Matches (Add back to unmatched lists for later phases)
    if len(prune_matched_idxs) > 0:
        if VERBOSE:
            print(
                f"CC Filtering: Kept {len(matched_idxs)} matches in largest component (Label {largest_label})"
            )
            print(f"Pruned {len(prune_matched_idxs)} matches in matching phase 1")
        unmatched_pred_idxs = np.concatenate(
            [unmatched_pred_idxs, prune_matched_idxs[:, 0]], axis=0
        ).astype(int)
        unmatched_gt_idxs = np.concatenate(
            [unmatched_gt_idxs, prune_matched_idxs[:, 1]], axis=0
        ).astype(int)
    return matched_idxs, unmatched_pred_idxs, unmatched_gt_idxs


def filter_matches_opening(
    matched_ori_gt_idxs, matched_idxs, unmatched_pred_idxs, unmatched_gt_idxs
):
    """
    Filters matches by applying Morphological Opening to remove isolated 'noise' matches
    while preserving clusters of matches.
    """
    # 1. Create Grid Mask
    matched_mask = np.zeros((NUM_GRID_KPT,), dtype="uint8")
    matched_mask[matched_ori_gt_idxs] = 1
    matched_mask = matched_mask.reshape(43, 55)

    # 2. Define Kernel
    # A 3x3 square kernel checks 8-connectivity (neighbors in all directions).
    # Isolated points (or very small clusters smaller than the kernel) will be removed.
    kernel = np.ones((3, 3), dtype=np.uint8)

    # 3. Apply Morphological Opening (Erosion -> Dilation)
    # This removes isolated points ("noise") but restores the boundary of larger clusters.
    # If you strictly want to count neighbors, Opening is the robust morphological equivalent.
    clean_mask = cv2.morphologyEx(matched_mask, cv2.MORPH_OPEN, kernel)

    # 4. Identify Valid Matches
    # Flatten the mask back to 1D to check against indices
    clean_mask_flat = clean_mask.flatten()

    # Check if each existing match survives the operation
    # (i.e., is the value at that index still 1?)
    is_valid = clean_mask_flat[matched_ori_gt_idxs] == 1

    # 5. Filter
    prune_matched_idxs = matched_idxs[~is_valid]
    matched_idxs = matched_idxs[is_valid]

    # 6. Recycle Pruned Matches
    if len(prune_matched_idxs) > 0:
        if VERBOSE:
            print(
                f"Opening Filtering: Kept {len(matched_idxs)} matches in dense clusters."
            )
            print(f"Pruned {len(prune_matched_idxs)} isolated matches.")

        unmatched_pred_idxs = np.concatenate(
            [unmatched_pred_idxs, prune_matched_idxs[:, 0]]
        ).astype(int)
        unmatched_gt_idxs = np.concatenate(
            [unmatched_gt_idxs, prune_matched_idxs[:, 1]]
        ).astype(int)

    return matched_idxs, unmatched_pred_idxs, unmatched_gt_idxs


def filter_matches_min_connections(
    matched_ori_gt_idxs,
    matched_idxs,
    unmatched_pred_idxs,
    unmatched_gt_idxs,
    min_connections=3,
):
    """
    Filters matches, keeping only those that have at least 'min_connections'
    neighbors (8-connectivity) in the grid.
    """
    # 1. Create Grid Mask
    matched_mask = np.zeros((NUM_GRID_KPT,), dtype="float32")  # float for filter2D
    matched_mask[matched_ori_gt_idxs] = 1
    matched_mask = matched_mask.reshape(43, 55)

    # 2. Define Neighbor-Counting Kernel
    # This kernel sums all 8 neighbors but ignores the center pixel itself.
    # [1, 1, 1]
    # [1, 0, 1]
    # [1, 1, 1]
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0

    # 3. Count Neighbors using Convolution
    # borderType=cv2.BORDER_CONSTANT assumes 0 neighbors outside image bounds
    neighbor_counts = cv2.filter2D(
        matched_mask, -1, kernel, borderType=cv2.BORDER_CONSTANT
    )

    # 4. Determine Valid Points
    # A point is valid IF:
    #   a. It is a match (matched_mask == 1)
    #   b. It has enough neighbors (neighbor_counts >= min_connections)
    is_valid_mask = (matched_mask == 1) & (neighbor_counts >= min_connections)

    # 5. Filter Indices
    # Flatten back to 1D to align with input indices
    is_valid_flat = is_valid_mask.flatten()

    # Check validity of existing matches
    keep_indices = is_valid_flat[matched_ori_gt_idxs]

    # Split
    prune_matched_idxs = matched_idxs[~keep_indices]
    matched_idxs = matched_idxs[keep_indices]

    # 6. Recycle Pruned Matches
    if len(prune_matched_idxs) > 0:
        if VERBOSE:
            print(
                f"Connection Filter: Kept {len(matched_idxs)} matches with >={min_connections} neighbors."
            )
            print(f"Pruned {len(prune_matched_idxs)} sparse matches.")

        unmatched_pred_idxs = np.concatenate(
            [unmatched_pred_idxs, prune_matched_idxs[:, 0]]
        ).astype(int)
        unmatched_gt_idxs = np.concatenate(
            [unmatched_gt_idxs, prune_matched_idxs[:, 1]]
        ).astype(int)

    return matched_idxs, unmatched_pred_idxs, unmatched_gt_idxs


def filter_matches_connected_to_matched_mask(
    base_matched_ori_gt_idxs,
    matched_ori_gt_idxs,
    matched_idxs,
    unmatched_pred_idxs,
    unmatched_gt_idxs,
):
    """
    Filters 'matched_idxs' by keeping only those that belong to connected components
    anchored to the structure defined by 'base_matched_ori_gt_idxs'.
    """
    # 1. Construct Base Mask from indices
    base_mask = np.zeros((NUM_GRID_KPT,), dtype="uint8")
    if len(base_matched_ori_gt_idxs) > 0:
        base_mask[base_matched_ori_gt_idxs] = 1
    base_mask = base_mask.reshape(43, 55)

    # 2. Construct Current Mask from indices
    current_mask = np.zeros((NUM_GRID_KPT,), dtype="uint8")
    if len(matched_ori_gt_idxs) > 0:
        current_mask[matched_ori_gt_idxs] = 1
    current_mask = current_mask.reshape(43, 55)

    # 3. Create Union Mask (Base U Current)
    # If a new match is adjacent to the base structure, they become one component here.
    union_mask = cv2.bitwise_or(current_mask, base_mask)

    # 4. Find Connected Components on the Union
    # connectivity=8 ensures diagonal neighbors are considered connected
    num_labels, labels = cv2.connectedComponents(union_mask, connectivity=4)

    # 5. Identify Anchors
    # Find which component labels are present in the "base" region.
    # Any new match belonging to these labels is valid (connected to base).
    anchored_labels = np.unique(labels[base_mask == 1])
    anchored_labels = anchored_labels[anchored_labels != 0]  # Remove background

    # 6. Filter Matches
    if len(anchored_labels) > 0:
        labels_flat = labels.flatten()

        # Check if the component ID of each new match is in the anchored list
        match_labels = labels_flat[matched_ori_gt_idxs]
        is_anchored = np.isin(match_labels, anchored_labels)

        # Split into kept and pruned
        prune_matched_idxs = matched_idxs[~is_anchored]
        matched_idxs = matched_idxs[is_anchored]
    else:
        # Edge case: Base mask was empty or disjoint from everything
        prune_matched_idxs = matched_idxs
        matched_idxs = np.empty((0, 2), dtype=int)

    # 7. Recycle Pruned Matches (Return them to the unmatched pool)
    if len(prune_matched_idxs) > 0:
        if VERBOSE:
            print(
                f"Union Filtering: Pruned {len(prune_matched_idxs)} isolated matches."
            )

        # Add pruned matches back to the unmatched arrays
        unmatched_pred_idxs = np.concatenate(
            [unmatched_pred_idxs, prune_matched_idxs[:, 0]]
        ).astype(int)
        unmatched_gt_idxs = np.concatenate(
            [unmatched_gt_idxs, prune_matched_idxs[:, 1]]
        ).astype(int)

    return matched_idxs, unmatched_pred_idxs, unmatched_gt_idxs


FOLD = 4
PREDICTION_PATH = f"/home/dangnh36/datasets/ecg/processed/pseudo_label/round1_finetune_dsnt/fold{FOLD}_predictions.pkl"
PREDICTION_DIR = os.path.dirname(PREDICTION_PATH)

with open(
    PREDICTION_PATH,
    "rb",
) as f:
    data = pickle.load(f)
print("PREDICTION KEYS:", data.keys())


MAIN_KPT_METHOD = "dsnt"
gt_df = data["gt_df"]
all_main_kpt_pred = data["main_kpt_preds"][MAIN_KPT_METHOD]
all_grid_kpt_pred = data["grid_kpt_pred"]
print(all_main_kpt_pred.shape, len(all_grid_kpt_pred))


with open(
    "/home/dangnh36/datasets/ecg/processed/reference_keypoints.json", "r"
) as f:
    ref_kpts = json.load(f)
ref_kpt_xys = np.array(list(ref_kpts.values()))
ref_kpt_names = list(ref_kpts.keys())
len(ref_kpt_names), ref_kpt_xys.shape


STD_REF_GRID_KPT_XYS = ref_kpt_xys[:2365]
_min_x = STD_REF_GRID_KPT_XYS[:, 0].min()
_max_x = STD_REF_GRID_KPT_XYS[:, 0].max()
_min_y = STD_REF_GRID_KPT_XYS[:, 1].min()
_max_y = STD_REF_GRID_KPT_XYS[:, 1].max()

MARGIN = 30
STD_REF_SAFE_POLYGON = np.array(
    [
        [_min_x - MARGIN, _min_y - MARGIN],
        [_max_x + MARGIN, _min_y - MARGIN],
        [_max_x + MARGIN, _max_y + MARGIN],
        [_min_x - MARGIN, _max_y + MARGIN],
    ],
    dtype=np.float32,
)
STD_REF_SAFE_POLYGON


### LOAD A STANDARD REFERENCE IMAGE (SHAPE 1700x2200)
REF_IMG = load_img(gt_df[0].to_dicts()[0])
print("REFERENCE IMAGE HAS SHAPE", REF_IMG.shape)

NUM_GRID_KPT = 2365
SKIP_MATCHED_SIZE = False
FILTER_OUTSIDE = False
CONF_THRES = 0.05
MATCH_FUNCTION = hungarian_match
# MATCH_FUNCTION = greedy_match
DIST_THRES1 = 15  # 15
DIST_THRES2 = 8
# COST_GATING1 = None  # 40 * sqrt(2)
COST_GATING1 = 28.3
COST_GATING2 = 28.3  # 20 * sqrt(2)

VIZ = False
VERBOSE = False

all_final_grid_kpt = np.zeros((len(gt_df), 2365, 2), dtype=np.float32)

for i in tqdm(range(len(gt_df))):
    # placeholder of 2365 grid keypoints for this sample
    final_grid_kpt = np.zeros_like(STD_REF_GRID_KPT_XYS)

    sample = gt_df[i].to_dicts()[0]
    img = None

    # fold 0: 106, 125
    # fold 1: 447, 1762
    # fold 2: 1284, 1434
    # fold 3:  22, 69, 105, 97, 170, 336, 519, 615, 633, 1095, 1011, 1492, 1731, 843, 1078
    # fold 4: 1104

    # DEBUG_IDX = 447
    # if i != DEBUG_IDX:
    #     continue

    grid_kpt = all_grid_kpt_pred[i]

    if VERBOSE:
        print(sample)
        print(
            "original:",
            i,
            grid_kpt.shape,
            grid_kpt[:, 2].min(),
            grid_kpt[:, 2].mean(),
            grid_kpt[:, 2].max(),
        )

    # ADDTIONAL FILTER BY CONFIDENT SCORE (already pre-filtered by 0.05)
    if len(grid_kpt) > NUM_GRID_KPT and CONF_THRES is not None:
        # print('before thresholding:', i, grid_kpt.shape, grid_kpt[:, 2].min(), grid_kpt[:, 2].mean(), grid_kpt[:, 2].max())
        if CONF_THRES == "top":
            sorted_idxs = np.argsort(-grid_kpt[:, 2])
            grid_kpt = grid_kpt[sorted_idxs[:NUM_GRID_KPT]]
        else:
            grid_kpt = grid_kpt[grid_kpt[:, 2] > CONF_THRES]
    if len(grid_kpt) == NUM_GRID_KPT and SKIP_MATCHED_SIZE:
        raise NotImplementedError("Need to infer the keypoints order")
        all_final_grid_kpt[i] = grid_kpt[:, :2]
        continue
    else:
        # print('after thresholding:', i, grid_kpt.shape, grid_kpt[:, 2].min(), grid_kpt[:, 2].mean(), grid_kpt[:, 2].max())
        pass

    # map back to ref coordinates
    if sample["H"] is None:
        ori_scale_x = ori_scale_y = 1.0
        ref_grid_kpt = grid_kpt.copy()[:, :2]
    else:
        to_ref_H = np.array(eval(sample["H"]))
        from_ref_H = np.linalg.pinv(to_ref_H)
        # roughtly estimate the scale from current original image (000x) relative to the reference image 0001
        ori_scale_x, ori_scale_y = get_scale_xy_from_homo_mat(from_ref_H)
        ref_grid_kpt = batch_perspective_transform_2d(
            grid_kpt[None, :, :2], to_ref_H[None]
        )[0]

    # filter outside keypoints
    if FILTER_OUTSIDE:
        outside_mask = check_points_inside_polygon(ref_grid_kpt, STD_REF_SAFE_POLYGON)
        # print('after outside:', i, grid_kpt.shape, grid_kpt[:, 2].min(), grid_kpt[:, 2].mean(), grid_kpt[:, 2].max())
        ref_grid_kpt = ref_grid_kpt[outside_mask]
        grid_kpt = grid_kpt[outside_mask]

    if len(grid_kpt) == NUM_GRID_KPT and SKIP_MATCHED_SIZE:
        raise NotImplementedError("Need to infer the keypoints order")
        all_final_grid_kpt[i] = grid_kpt[:, :2]
        continue

    matched_idxs, unmatched_pred_idxs, unmatched_gt_idxs = MATCH_FUNCTION(
        ref_grid_kpt,
        grid_kpt[:, 2],
        STD_REF_GRID_KPT_XYS,
        threshold=DIST_THRES1,
        cost_gating=COST_GATING1,
    )

    # ===== FILTER LARGEST CONNECTED COMPONENT =====
    # filter matched one inside a largest connected component
    # this helps filter some FP matches
    # for example, sample 105 of fold 3
    matched_idxs, unmatched_pred_idxs, unmatched_gt_idxs = filter_matches_largest_cca(
        matched_idxs[:, 1], matched_idxs, unmatched_pred_idxs, unmatched_gt_idxs
    )

    # remove "isolated" matched (which usually a wrongly matched keypoints)
    matched_idxs, unmatched_pred_idxs, unmatched_gt_idxs = (
        filter_matches_min_connections(
            matched_idxs[:, 1],
            matched_idxs,
            unmatched_pred_idxs,
            unmatched_gt_idxs,
            min_connections=4,
        )
    )

    if VERBOSE:
        print(
            "match1",
            len(matched_idxs),
            len(unmatched_pred_idxs),
            len(unmatched_gt_idxs),
        )

    # ===== VISUALIZE =====
    if VIZ:
        img = load_img(sample)

        viz = visualize_matches(
            img,
            matched_idxs,
            unmatched_pred_idxs,
            unmatched_gt_idxs,
            grid_kpt[:, :2],
            None,
        )
        display(Image.fromarray(viz))

        viz = visualize_matches(
            REF_IMG,
            matched_idxs,
            unmatched_pred_idxs,
            unmatched_gt_idxs,
            ref_grid_kpt,
            STD_REF_GRID_KPT_XYS,
        )
        display(Image.fromarray(viz))
    # ======= END VISUALIZE =====

    # format: pred_idx, gt_idx, pred_x, pred_y, gt_x, gt_y
    matches = np.concatenate(
        [
            matched_idxs,
            grid_kpt[matched_idxs[:, 0], :2],
            STD_REF_GRID_KPT_XYS[matched_idxs[:, 1]],
        ],
        axis=1,
    )
    # format: x, y, index
    if len(unmatched_gt_idxs) > 0:
        unmatched_gts = np.concatenate(
            [STD_REF_GRID_KPT_XYS[unmatched_gt_idxs], unmatched_gt_idxs[:, None]],
            axis=1,
        )
    else:
        unmatched_gts = np.empty((0, 3), dtype="float32")

    # format: x, y, conf, index
    if len(unmatched_pred_idxs) > 0:
        unmatched_preds = np.concatenate(
            [grid_kpt[unmatched_pred_idxs], unmatched_pred_idxs[:, None]], axis=1
        )
    else:
        unmatched_preds = np.empty((0, 3), dtype="float32")

    # ===== matched in phase 1 =====
    if len(matches):
        final_grid_kpt[matches[:, 1].astype(np.uint32)] = matches[:, 2:4]
    phase1_num_matches = len(matches)

    # ========== INTERATIVELY HANDLE UNMATCHED GT =========

    cur_round = 0
    all_matched_ori_gt_idxs2 = []
    unmatched2_ori_gt_idxs = []
    while True:
        if len(unmatched_gts) == 0:
            break
        cur_round += 1
        interpolated_xys = []
        for j, ref_gt_point in enumerate(unmatched_gts):
            # _viz_interpolate = False
            _viz_interpolate = VIZ and (cur_round == -1)
            # find top 24 nearest GT points
            cur_gt_xy = ref_gt_point[:2]
            # cur_gt_idx = ref_gt_point[2]
            dists = cur_gt_xy[None] - matches[:, 4:6]
            dists = (dists[:, 0] ** 2 + dists[:, 1] ** 2) ** 0.5
            sort_idxs = np.argsort(dists)
            # print(dists.shape, sort_idxs.shape)
            nearest_idxs = sort_idxs[:24]
            nearest_matches = matches[nearest_idxs]

            # find local HOMO TRANSFORMATION MATRIX from REF TO CURRENT (PREDICTION)
            src_pts = nearest_matches[:, 4:6]
            dst_pts = nearest_matches[:, 2:4]
            # ransacReprojThreshold: Even though MAGSAC is "threshold-free", OpenCV still requires
            # this parameter as an upper bound for internal optimizations. 3.0 - 5.0 is standard.
            local_H, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.USAC_MAGSAC,  # MAGSAC++ algorithm
                ransacReprojThreshold=8.0,
                maxIters=5000,
                confidence=0.9999,
            )
            if mask.sum() < mask.size:
                # print(
                #     "WARNING: local homo matrix estimation resulted in unmatched pairs",
                #     mask.sum(),
                #     "/",
                #     mask.size,
                # )
                pass

            interpolated_pred_xy = cv2.perspectiveTransform(
                cur_gt_xy.reshape(1, 1, 2), local_H
            )[0, 0]
            interpolated_xys.append(interpolated_pred_xy)

            if _viz_interpolate:
                # === IMPROVED VISUALIZATION SNIPPET START ===
                # 1. Setup Canvas
                # Ensure 'img' is the current input image (RGB)
                vis_ref = REF_IMG.copy()
                vis_pred = img.copy()

                center_ref = tuple(cur_gt_xy.astype(int))
                # Ensure interpolated coordinates are integers for OpenCV drawing
                center_pred = tuple(interpolated_pred_xy.astype(int))

                # DEBUG PRINT: Check if coordinates are crazy
                print(f"--- DEBUG Unmatched GT Index {j} ---")
                print(f"Reference Point: {center_ref}")
                print(f"Interpolated Prediction Point: {center_pred}")
                print(f"Prediction Image Shape: {vis_pred.shape}")

                # Check if out of bounds
                if (
                    center_pred[0] < 0
                    or center_pred[0] >= vis_pred.shape[1]
                    or center_pred[1] < 0
                    or center_pred[1] >= vis_pred.shape[0]
                ):
                    print(
                        "WARNING: Interpolated point is OUT OF BOUNDS on the prediction image!"
                    )

                # 2. Draw on Reference (Source of Truth) - LEFT IMAGE
                for pt in src_pts.astype(int):
                    # Yellow lines connecting target to anchors
                    cv2.line(vis_ref, center_ref, tuple(pt), (255, 255, 0), 1)
                    # Green anchor points
                    cv2.circle(vis_ref, tuple(pt), 5, (0, 255, 0), -1)
                # Red Target (Missing GT)
                cv2.circle(vis_ref, center_ref, 10, (255, 0, 0), -1)

                # 3. Draw on Prediction (Interpolated Result) - RIGHT IMAGE
                for pt in dst_pts.astype(int):
                    # Yellow lines
                    cv2.line(vis_pred, center_pred, tuple(pt), (255, 255, 0), 1)
                    # Green anchor points (detected by model)
                    cv2.circle(vis_pred, tuple(pt), 5, (0, 255, 0), -1)
                # Red Result (Recovered via interpolation)
                cv2.circle(vis_pred, center_pred, 10, (255, 0, 0), -1)

                # 4. Stack Side-by-Side (Auto-resize pred to match ref height)
                h_ref = vis_ref.shape[0]
                scale_factor = h_ref / vis_pred.shape[0]
                # Resize pred to match reference height for clean stacking
                vis_pred_resized = cv2.resize(
                    vis_pred, (int(vis_pred.shape[1] * scale_factor), h_ref)
                )
                combined_view = np.hstack([vis_ref, vis_pred_resized])

                # 5. Display FULL VIEW (No cropping to ensure visibility)
                # Downscale slightly for browser display if it's huge
                if combined_view.shape[0] > 1200:
                    scale_disp = 1200 / combined_view.shape[0]
                    combined_view = cv2.resize(
                        combined_view, (0, 0), fx=scale_disp, fy=scale_disp
                    )

                display(Image.fromarray(combined_view))
                print("------------------------------------")
                # === IMPROVED VISUALIZATION SNIPPET END ===

        interpolated_xys = np.array(interpolated_xys)
        # print('HEY', interpolated_xys.shape)

        # ================== MATCH INTERPOLATED POINTS WITH MODEL'S PREDICTED POINTS =============
        thres2 = DIST_THRES2 * ((ori_scale_x**2 + ori_scale_y**2) ** 0.5) / (2**0.5)
        if COST_GATING2 is not None:
            cost_gating2 = (
                COST_GATING2 * ((ori_scale_x**2 + ori_scale_y**2) ** 0.5) / (2**0.5)
            )
        else:
            cost_gating2 = None
        matched_idxs2, unmatched_pred_idxs2, unmatched_gt_idxs2 = MATCH_FUNCTION(
            unmatched_preds[:, :2],
            unmatched_preds[:, 2],
            interpolated_xys,
            threshold=thres2,
            cost_gating=cost_gating2,
        )

        if len(matched_idxs2) > 0:
            # matched_idxs2, unmatched_pred_idxs2, unmatched_gt_idxs2 = filter_matches_largest_cca(
            #     unmatched_gts[matched_idxs2[:, 1], 2].astype(int),
            #     matched_idxs2, unmatched_pred_idxs2, unmatched_gt_idxs2
            # )

            matched_idxs2, unmatched_pred_idxs2, unmatched_gt_idxs2 = (
                filter_matches_connected_to_matched_mask(
                    matches[:, 1].astype(int),
                    unmatched_gts[matched_idxs2[:, 1], 2].astype(int),
                    matched_idxs2,
                    unmatched_pred_idxs2,
                    unmatched_gt_idxs2,
                )
            )

        # FOR DEBUG PURPOSE ONLY, DON'T COMMENT OUT
        # matched_idxs2 = []
        # unmatched_pred_idxs2 = []
        # unmatched_gt_idxs2 = np.arange(0, len(interpolated_xys))

        if VERBOSE:
            print(
                "match2",
                len(matched_idxs2),
                len(unmatched_pred_idxs2),
                len(unmatched_gt_idxs2),
            )

        # matched in phase 2
        if len(matched_idxs2) > 0:
            matched_ori_gt_idxs2 = unmatched_gts[matched_idxs2[:, 1], 2].astype(int)
            matched_ori_pred_idxs2 = unmatched_preds[matched_idxs2[:, 0], 3].astype(int)
            final_grid_kpt[matched_ori_gt_idxs2] = unmatched_preds[
                matched_idxs2[:, 0], :2
            ]
            if VIZ:
                print("VIZ!")
                src = np.zeros((2365,), dtype=int)
                src[matches[:, 1].astype(np.uint32)] = 0
                src[matched_ori_gt_idxs2] = 1  # ORANGE
                viz_img_keypoints(
                    img,
                    kpt_xys=final_grid_kpt,
                    kpt_classes=src,
                    save_path=None,
                    figsize=(12, 12),
                    point_size=10,
                    alpha=0.9,
                    show_legend=False,
                )

            # extend matches: pred_idx, gt_idx, pred_x, pred_y, gt_x, gt_y
            new_matches = np.concatenate(
                [
                    matched_ori_pred_idxs2[:, None],
                    matched_ori_gt_idxs2[:, None],
                    unmatched_preds[matched_idxs2[:, 0], :2],
                    unmatched_gts[matched_idxs2[:, 1], :2],
                ],
                axis=1,
            )
            matches = np.concatenate([matches, new_matches], axis=0)
        else:
            matched_ori_gt_idxs2 = []
            matched_ori_pred_idxs2 = []

        # reduce unmatched_gts and unmatched_preds
        if len(unmatched_gt_idxs2) > 0:
            # print('REDUCE UNMATCHED GT BEFORE:', unmatched_gts[:, 2].astype(int))
            unmatched2_ori_gt_idxs = unmatched_gts[unmatched_gt_idxs2, 2].astype(int)
            unmatched_gts = unmatched_gts[unmatched_gt_idxs2]
            # print('AFTER:', unmatched_gts[:, 2].astype(int))
        else:
            unmatched2_ori_gt_idxs = []
            unmatched_gts = np.empty((0, 3), dtype="float32")

        if len(unmatched_pred_idxs2) > 0:
            unmatched_preds = unmatched_preds[unmatched_pred_idxs2]
        else:
            unmatched_preds = np.empty((0, 3), dtype="float32")

        all_matched_ori_gt_idxs2.extend(matched_ori_gt_idxs2)

        # if can not find any matched pair, stop
        if len(matched_ori_gt_idxs2) == 0:
            # print("Can not find any matched pairs, Stop.")
            break
        if VERBOSE:
            print(
                f"Iterative round {cur_round}: matches={len(all_matched_ori_gt_idxs2)}(+{len(matched_ori_gt_idxs2)}) remain_preds={len(unmatched_preds)} remain_gts={len(unmatched_gts)} "
            )

    assert (
        len(unmatched_gts) == 0 or len(matched_ori_gt_idxs2) == 0
    ), f"{len(unmatched_gts)} {len(matches)}"

    if VERBOSE:
        print(
            f"matches={len(matches)} unmatched_preds={len(unmatched_preds)} unmatched_gt={len(unmatched_gt_idxs2)}"
        )

    # unmatched GT, use interpolated points
    if len(unmatched2_ori_gt_idxs) > 0:
        if VERBOSE:
            print(
                f"Interpolate {len(unmatched2_ori_gt_idxs)} points:",
                unmatched2_ori_gt_idxs,
            )
        # print(interpolated_xys, unmatched_gt_idxs2)
        final_grid_kpt[unmatched2_ori_gt_idxs] = interpolated_xys[unmatched_gt_idxs2]

    assert (
        phase1_num_matches + len(all_matched_ori_gt_idxs2) + len(unmatched2_ori_gt_idxs)
        == NUM_GRID_KPT
    )
    assert not np.any(np.all(final_grid_kpt == 0.0, axis=1), axis=0)

    if VIZ:
        # if len(unmatched2_ori_gt_idxs) > 0:
        print(
            f"{i} FINAL VIZ: matches={len(matches)} unmatched_preds={len(unmatched_preds)} unmatched_gt={len(unmatched_gt_idxs2)}"
        )
        if img is None:
            img = load_img(sample)
        src = np.zeros((2365,), dtype=int)
        src[matches[:, 1].astype(np.uint32)] = 0
        src[unmatched2_ori_gt_idxs] = 1  # orange
        viz_img_keypoints(
            img,
            kpt_xys=final_grid_kpt,
            kpt_classes=src,
            save_path=None,
            figsize=(12, 12),
            point_size=10,
            alpha=0.9,
            show_legend=False,
        )

    all_final_grid_kpt[i] = final_grid_kpt

    # break

    # if i >= 100:
    #     break

print("FINAL", all_final_grid_kpt.shape, all_main_kpt_pred.shape)
final_pred = np.concatenate([all_final_grid_kpt, all_main_kpt_pred], axis=1)
print(final_pred.shape)
