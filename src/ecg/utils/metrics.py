import numpy as np
import time
from tqdm import tqdm
import numpy as np
from numba import njit
from joblib import Parallel, delayed


def keypoint_accuracy(
    pred,
    gt,
    threshold,
    threshold_factors=None,
):
    """
    Compute keypoint accuracy (PCK-style) with flexible shapes.

    Args:
        pred (np.ndarray): (N, C, 2) or (N, 2)
        gt (np.ndarray):   (N, C, 2) or (N, 2)
        threshold (float or np.ndarray):
            Scalar or array broadcastable to (N, C)
        threshold_factors (array-like or None):
            If None, defaults to np.arange(0.5, 1.01, 0.05)

    Returns:
        acc_per_factor (np.ndarray): (T,) accuracy for each threshold factor
        avg_acc (float): average accuracy over all threshold factors
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)

    assert pred.shape == gt.shape, f"{pred.shape}, {gt.shape}"
    assert pred.shape[-1] == 2

    # Handle (N, 2) → (N, 1, 2)
    if pred.ndim == 2:
        pred = pred[:, None, :]
        gt = gt[:, None, :]

    if threshold_factors is None:
        threshold_factors = np.arange(0.5, 1.01, 0.05)

    threshold_factors = np.asarray(threshold_factors)

    # Euclidean distances: (N, C)
    dists = np.linalg.norm(pred - gt, axis=-1)

    # Prepare threshold for broadcasting: (N, C)
    threshold = np.asarray(threshold)
    if threshold.ndim == 1:
        threshold = threshold[:, None]  # (N,) -> (N, 1) to be broadcastable
    threshold = np.broadcast_to(threshold, dists.shape)

    # Expand for threshold factors
    # dists:      (N, C, 1)
    # thresholds: (N, C, T)
    dists = dists[..., None]
    thresholds = threshold[..., None] * threshold_factors[None, None, :]

    # Correct predictions
    correct = dists <= thresholds

    # Accuracy per threshold factor
    acc_per_factor = correct.mean(axis=(0, 1))

    # Average accuracy
    avg_acc = float(acc_per_factor.mean())

    return acc_per_factor, avg_acc


def keypoint_detection_map_slow(
    pred,
    gt,
    threshold,
    threshold_factors=None,
):
    """
    Compute Object-Detection–style mean Average Precision (mAP) for
    INDEPENDENT keypoint detection.

    Each keypoint is treated as a standalone detection (no class labels).
    For multi-class keypoints (e.g., Nose, Elbow), call this function
    separately per class.

    Metric definition (per threshold factor):
        1. Collect all predictions across the dataset.
        2. Sort predictions globally by confidence (descending).
        3. Greedy matching:
            - Matching is restricted to the same image.
            - Each GT keypoint can be matched at most once.
            - Each prediction matches the nearest unmatched GT.
            - A match is TP if distance <= threshold[img] * factor, else FP.
        4. Build a Precision–Recall curve.
        5. Compute AP using COCO-style PR integration.
    mAP is the mean AP over all threshold factors.

    Args:
        pred (list[np.ndarray]):
            List of length N. Each element has shape (Pi, 3) where
            columns are (x, y, confidence).

        gt (list[np.ndarray]):
            List of length N. Each element has shape (Gi, 2) containing
            ground-truth keypoint coordinates.

        threshold (float or np.ndarray):
            Base Euclidean distance threshold in pixels.
            Can be a scalar or an array of shape (N,) specifying
            per-image thresholds.

        threshold_factors (array-like or None):
            Multiplicative factors applied to `threshold`.
            If None, defaults to np.arange(0.5, 1.01, 0.1).

    Returns:
        dict with:
            - 'ap_per_factor': np.ndarray of shape (T,)
            - 'map': float (mean AP over all threshold factors)
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    N = len(pred)
    assert len(gt) == N, "pred and gt must have the same length"

    if threshold_factors is None:
        threshold_factors = np.arange(0.5, 1.01, 0.1)
    threshold_factors = np.asarray(threshold_factors, dtype=float)

    threshold = np.asarray(threshold, dtype=float)
    if threshold.ndim == 0:
        threshold = np.full(N, threshold.item(), dtype=float)
    elif threshold.ndim == 2:
        assert threshold.shape[1] == 1
        threshold = threshold[:, 0]
    assert threshold.shape == (N,), f"threshold must be scalar or shape ({N},)"

    ap_per_factor = []

    # ------------------------------------------------------------------
    # Loop over threshold factors
    # ------------------------------------------------------------------
    for factor in threshold_factors:
        # Track matched GT keypoints per image
        gt_used = [np.zeros(len(g), dtype=bool) for g in gt]

        total_gt = sum(len(g) for g in gt)
        if total_gt == 0:
            ap_per_factor.append(0.0)
            continue

        # Collect all predictions globally: (confidence, image_id, pred_idx)
        all_preds = []
        for img_id in range(N):
            p = np.asarray(pred[img_id], dtype=float)

            if p.ndim != 2 or p.shape[1] != 3:
                raise ValueError(
                    f"pred[{img_id}] must have shape (P, 3), got {p.shape}"
                )

            if len(p) == 0:
                continue

            coords = p[:, :2]
            scores = p[:, 2]

            if not np.all(np.isfinite(scores)):
                raise ValueError(f"Non-finite confidence detected in image {img_id}")

            for j in range(len(coords)):
                all_preds.append((scores[j], img_id, j))

        # Sort predictions by confidence (descending)
        all_preds.sort(key=lambda x: -x[0])

        tp = []
        fp = []

        # --------------------------------------------------------------
        # Greedy confidence-ordered matching
        # --------------------------------------------------------------
        for conf, img_id, pred_idx in tqdm(all_preds, desc=f"factor={factor}"):
            g_coords = np.asarray(gt[img_id], dtype=float)

            # No GT in this image → FP
            if len(g_coords) == 0:
                tp.append(0)
                fp.append(1)
                continue

            pred_xy = np.asarray(pred[img_id], dtype=float)[pred_idx, :2]

            # Distance to all GT keypoints in the image
            dists = np.linalg.norm(pred_xy - g_coords, axis=1)

            valid = ~gt_used[img_id]
            if not np.any(valid):
                tp.append(0)
                fp.append(1)
                continue

            masked_dists = np.where(valid, dists, np.inf)
            min_idx = np.argmin(masked_dists)
            min_dist = masked_dists[min_idx]

            if min_dist <= threshold[img_id] * factor:
                gt_used[img_id][min_idx] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        # --------------------------------------------------------------
        # Precision–Recall and AP computation
        # --------------------------------------------------------------
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        recall = tp / total_gt
        precision = tp / np.maximum(tp + fp, 1e-12)

        # COCO-style padding
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([0.0], precision, [0.0]))

        # Precision envelope
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])

        # Area under PR curve
        ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
        ap_per_factor.append(ap)

    ap_per_factor = np.asarray(ap_per_factor, dtype=float)

    return ap_per_factor, float(ap_per_factor.mean())


@njit
def greedy_match(
    pred_coords,
    pred_img_ids,
    gt_coords,
    gt_used,
    threshold,
    factor,
):
    M = pred_coords.shape[0]
    tp = np.zeros(M, dtype=np.float32)
    fp = np.zeros(M, dtype=np.float32)

    for i in range(M):
        img_id = pred_img_ids[i]
        g = gt_coords[img_id]

        if g.shape[0] == 0:
            fp[i] = 1.0
            continue

        min_dist = 1e12
        min_j = -1

        for j in range(g.shape[0]):
            if gt_used[img_id][j]:
                continue

            dx = g[j, 0] - pred_coords[i, 0]
            dy = g[j, 1] - pred_coords[i, 1]
            d = (dx * dx + dy * dy) ** 0.5

            if d < min_dist:
                min_dist = d
                min_j = j

        if min_j >= 0 and min_dist <= threshold[img_id] * factor:
            gt_used[img_id][min_j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    return tp, fp


def keypoint_detection_map_numba(
    pred,
    gt,
    threshold,
    threshold_factors=None,
):
    if threshold_factors is None:
        threshold_factors = np.arange(0.5, 1.01, 0.1)
    threshold_factors = np.asarray(threshold_factors, dtype=float)

    N = len(pred)
    threshold = np.asarray(threshold, dtype=float)
    if threshold.ndim == 0:
        threshold = np.full(N, threshold.item(), dtype=float)

    # GT
    gt_coords = [np.asarray(g, dtype=np.float32) for g in gt]
    total_gt = sum(len(g) for g in gt_coords)
    if total_gt == 0:
        return np.zeros(len(threshold_factors)), 0.0

    # Flatten preds
    coords, img_ids, scores = [], [], []
    for img_id, p in enumerate(pred):
        if len(p) == 0:
            continue
        p = np.asarray(p, dtype=np.float32)
        coords.append(p[:, :2])
        scores.append(p[:, 2])
        img_ids.append(np.full(len(p), img_id, dtype=np.int32))

    pred_coords = np.concatenate(coords)
    pred_scores = np.concatenate(scores)
    pred_img_ids = np.concatenate(img_ids)

    order = np.argsort(-pred_scores)
    pred_coords = pred_coords[order]
    pred_img_ids = pred_img_ids[order]

    ap_per_factor = np.zeros(len(threshold_factors))

    for i, factor in enumerate(threshold_factors):
        gt_used = [np.zeros(len(g), dtype=np.bool_) for g in gt_coords]

        tp, fp = greedy_match(
            pred_coords,
            pred_img_ids,
            gt_coords,
            gt_used,
            threshold,
            factor,
        )

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        recall = tp / total_gt
        precision = tp / np.maximum(tp + fp, 1e-12)

        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([0.0], precision, [0.0]))
        precision = np.maximum.accumulate(precision[::-1])[::-1]

        ap_per_factor[i] = np.sum((recall[1:] - recall[:-1]) * precision[1:])

    return ap_per_factor, float(ap_per_factor.mean())


@njit
def match_single_image(pred_xy, pred_score, gt_xy, threshold):
    """
    Greedy confidence-ordered matching for ONE image.
    """
    P = pred_xy.shape[0]
    G = gt_xy.shape[0]

    order = np.argsort(-pred_score)
    pred_xy = pred_xy[order]
    pred_score = pred_score[order]

    tp = np.zeros(P, dtype=np.float32)
    fp = np.zeros(P, dtype=np.float32)
    gt_used = np.zeros(G, dtype=np.bool_)

    for i in range(P):
        if G == 0:
            fp[i] = 1.0
            continue

        min_dist = 1e12
        min_j = -1

        for j in range(G):
            if gt_used[j]:
                continue
            dx = gt_xy[j, 0] - pred_xy[i, 0]
            dy = gt_xy[j, 1] - pred_xy[i, 1]
            d = (dx * dx + dy * dy) ** 0.5
            if d < min_dist:
                min_dist = d
                min_j = j

        if min_j >= 0 and min_dist <= threshold:
            gt_used[min_j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    return pred_score, tp, fp


def keypoint_detection_map_parallel(
    pred,
    gt,
    threshold,
    threshold_factors=None,
    n_jobs=-1,
):
    if threshold_factors is None:
        threshold_factors = np.arange(0.5, 1.01, 0.1)
    threshold_factors = np.asarray(threshold_factors, dtype=float)

    N = len(pred)

    threshold = np.asarray(threshold, dtype=float)
    if threshold.ndim == 0:
        threshold = np.full(N, threshold.item(), dtype=float)

    gt_coords = [np.asarray(g, dtype=np.float32) for g in gt]
    total_gt = sum(len(g) for g in gt_coords)

    if total_gt == 0:
        return np.zeros(len(threshold_factors)), 0.0

    ap_per_factor = np.zeros(len(threshold_factors), dtype=float)

    for t_idx, factor in enumerate(threshold_factors):
        results = Parallel(n_jobs=n_jobs)(
            delayed(match_single_image)(
                np.asarray(pred[i][:, :2], dtype=np.float32),
                np.asarray(pred[i][:, 2], dtype=np.float32),
                gt_coords[i],
                threshold[i] * factor,
            )
            for i in range(N)
            if len(pred[i]) > 0
        )

        if not results:
            continue
        
        # A HARD BUG was there, affect AP calculation when number of images get larger, typically over ~7000 samples
        # original dtype is float16, which is overflow in subsequent operations,
        # resulting in wrong AP calculation
        # .astype('float64') is a MUST to fix the metric bug
        scores = np.concatenate([r[0] for r in results]).astype('float64')
        tp = np.concatenate([r[1] for r in results]).astype('float64')
        fp = np.concatenate([r[2] for r in results]).astype('float64')

        order = np.argsort(-scores)
        tp = tp[order]
        fp = fp[order]

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        recall = tp / total_gt
        precision = tp / np.maximum(tp + fp, 1e-12)

        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([0.0], precision, [0.0]))
        precision = np.maximum.accumulate(precision[::-1])[::-1]

        ap_per_factor[t_idx] = np.sum((recall[1:] - recall[:-1]) * precision[1:])

    return ap_per_factor, float(ap_per_factor.mean())


def compute_metrics(
    main_kpt_preds, grid_kpt_pred, kpt_gt, kptness_gt, per_img_threshold
):
    assert kpt_gt.shape[1:] == (2422, 2)
    assert kptness_gt.shape[1:] == (2422,)

    metrics = {}

    main_kpt_gt = kpt_gt[:, -57:]
    grid_kpt_gt = kpt_gt[:, :-57]
    grid_kptness_gt = kptness_gt[:, :-57]

    remain_grid_kpt_gt = []
    for img_grid_kpt_gt, img_grid_kptness_gt in zip(grid_kpt_gt, grid_kptness_gt):
        remain_grid_kpt_gt.append(img_grid_kpt_gt[img_grid_kptness_gt > 0])

    # all equal to number of images
    assert (
        len(grid_kpt_pred) == len(kpt_gt) == len(kptness_gt) == len(per_img_threshold)
    )

    for method_name, main_kpt_pred in main_kpt_preds.items():
        print(method_name)
        acc_per_factor, avg_acc = keypoint_accuracy(
            main_kpt_pred[..., :2],
            main_kpt_gt,
            per_img_threshold,
            threshold_factors=None,
        )
        print("MAIN ACC PER FACTOR:", acc_per_factor)
        metrics[f"{method_name}__AVG_ACC"] = avg_acc
        metrics[f"{method_name}__ACC"] = acc_per_factor[-1]

    _t0 = time.time()
    _ap_per_factor, avg_ap = keypoint_detection_map_parallel(
        grid_kpt_pred,
        remain_grid_kpt_gt,
        per_img_threshold,
        threshold_factors=None,
    )
    _t1 = time.time()
    print(f"Compute grid points AP take {_t1 - _t0:.2f} sec")
    print("GRID AP PER FACTOR:", _ap_per_factor)
    metrics["GRID_AP"] = avg_ap
    return metrics


# ====================================================
# ==================== TEST CASES ====================
# ====================================================
def _test_compute_metrics_perfect_prediction():
    """
    Perfect prediction:
    - main keypoints exactly match GT
    - grid keypoints exactly match GT
    Expected:
    - MAIN_ACC == 1.0
    - GRID_AP == 1.0
    """
    N = 2  # number of images
    NUM_KPTS = 2422
    NUM_MAIN = 57
    NUM_GRID = NUM_KPTS - NUM_MAIN

    # --------------------------------------------------
    # Ground truth
    # --------------------------------------------------
    kpt_gt = np.zeros((N, NUM_KPTS, 2), dtype=np.float32)

    # Simple deterministic coordinates
    for i in range(N):
        kpt_gt[i, :, 0] = np.arange(NUM_KPTS)
        kpt_gt[i, :, 1] = np.arange(NUM_KPTS)

    # All grid keypoints are valid
    kptness_gt = np.ones((N, NUM_KPTS), dtype=np.int32)

    # --------------------------------------------------
    # Main keypoint predictions (perfect)
    # --------------------------------------------------
    main_kpt_preds = {
        "methodA": kpt_gt[:, -NUM_MAIN:].copy(),
    }

    # --------------------------------------------------
    # Grid keypoint predictions (perfect)
    # Convert GT → detection format (x, y, conf)
    # --------------------------------------------------
    grid_kpt_pred = []
    for i in range(N):
        coords = kpt_gt[i, :-NUM_MAIN]
        conf = np.ones((coords.shape[0], 1), dtype=np.float32)
        grid_kpt_pred.append(np.concatenate([coords, conf], axis=1))

    # --------------------------------------------------
    # Thresholds (large enough to always match)
    # --------------------------------------------------
    per_img_threshold = np.full((N,), 1e-3)

    metrics = compute_metrics(
        main_kpt_preds=main_kpt_preds,
        grid_kpt_pred=grid_kpt_pred,
        kpt_gt=kpt_gt,
        kptness_gt=kptness_gt,
        per_img_threshold=per_img_threshold,
    )
    print(metrics)

    assert np.isclose(metrics["methodA__AVG_ACC"], 1.0)
    assert np.isclose(metrics["GRID_AP"], 1.0)


def _test_compute_metrics_grid_masking():
    """
    Half of grid keypoints are masked out.
    Predictions only include valid ones.
    AP should still be 1.0.
    """
    N = 1
    NUM_KPTS = 2422
    NUM_MAIN = 57
    NUM_GRID = NUM_KPTS - NUM_MAIN

    kpt_gt = np.zeros((N, NUM_KPTS, 2), dtype=np.float32)
    kpt_gt[0, :, 0] = np.arange(NUM_KPTS)
    kpt_gt[0, :, 1] = np.arange(NUM_KPTS)

    # Mask out half of grid keypoints
    kptness_gt = np.zeros((N, NUM_KPTS), dtype=np.int32)
    kptness_gt[0, :NUM_GRID:2] = 1  # keep every other grid keypoint
    kptness_gt[0, -NUM_MAIN:] = 1  # keep all main keypoints

    main_kpt_preds = {
        "methodA": kpt_gt[:, -NUM_MAIN:].copy(),
    }

    # Only include valid grid keypoints
    valid_grid = kpt_gt[0, :NUM_GRID][kptness_gt[0, :NUM_GRID] > 0]
    conf = np.ones((valid_grid.shape[0], 1), dtype=np.float32)
    grid_kpt_pred = [np.concatenate([valid_grid, conf], axis=1)]

    per_img_threshold = np.array([1e-3])

    metrics = compute_metrics(
        main_kpt_preds=main_kpt_preds,
        grid_kpt_pred=grid_kpt_pred,
        kpt_gt=kpt_gt,
        kptness_gt=kptness_gt,
        per_img_threshold=per_img_threshold,
    )
    print(metrics)

    assert np.isclose(metrics["methodA__AVG_ACC"], 1.0)
    assert np.isclose(metrics["GRID_AP"], 1.0)


def _test_compute_metrics_grid_all_wrong():
    """
    Grid predictions are far from GT → AP should be 0.
    """
    N = 1
    NUM_KPTS = 2422
    NUM_MAIN = 57
    NUM_GRID = NUM_KPTS - NUM_MAIN

    kpt_gt = np.zeros((N, NUM_KPTS, 2), dtype=np.float32)
    kptness_gt = np.ones((N, NUM_KPTS), dtype=np.int32)

    main_kpt_preds = {
        "methodA": kpt_gt[:, -NUM_MAIN:].copy(),
    }

    # Far-away grid predictions
    coords = kpt_gt[0, :NUM_GRID] + 1000.0
    conf = np.ones((coords.shape[0], 1), dtype=np.float32)
    grid_kpt_pred = [np.concatenate([coords, conf], axis=1)]

    per_img_threshold = np.array([1e-3])

    metrics = compute_metrics(
        main_kpt_preds=main_kpt_preds,
        grid_kpt_pred=grid_kpt_pred,
        kpt_gt=kpt_gt,
        kptness_gt=kptness_gt,
        per_img_threshold=per_img_threshold,
    )
    print(metrics)

    assert np.isclose(metrics["methodA__AVG_ACC"], 1.0)
    assert np.isclose(metrics["GRID_AP"], 0.0)


def _test_compute_metrics_with_noise():
    """
    Add Gaussian noise to predictions and verify that:
    - Metrics are < 1.0
    - Metrics are > 0.0
    - Larger noise leads to worse metrics
    """
    rng = np.random.default_rng(seed=42)

    N = 2
    NUM_KPTS = 2422
    NUM_MAIN = 57
    NUM_GRID = NUM_KPTS - NUM_MAIN

    # --------------------------------------------------
    # Ground truth
    # --------------------------------------------------
    kpt_gt = np.zeros((N, NUM_KPTS, 2), dtype=np.float32)
    for i in range(N):
        kpt_gt[i, :, 0] = np.arange(NUM_KPTS)
        kpt_gt[i, :, 1] = np.arange(NUM_KPTS)

    kptness_gt = np.ones((N, NUM_KPTS), dtype=np.int32)

    per_img_threshold = np.full(N, 2.0, dtype=np.float32)

    def run_with_noise(std):
        # ------------------------------
        # Main keypoints (noisy)
        # ------------------------------
        main_noise = rng.normal(0, std, size=(N, NUM_MAIN, 2))
        main_pred = kpt_gt[:, -NUM_MAIN:] + main_noise

        main_kpt_preds = {"methodA": main_pred}

        # ------------------------------
        # Grid keypoints (noisy detections)
        # ------------------------------
        grid_kpt_pred = []
        for i in range(N):
            coords = kpt_gt[i, :-NUM_MAIN]
            noise = rng.normal(0, std, size=coords.shape)
            noisy_coords = coords + noise

            conf = np.ones((coords.shape[0], 1), dtype=np.float32)
            grid_kpt_pred.append(np.concatenate([noisy_coords, conf], axis=1))

        metrics = compute_metrics(
            main_kpt_preds=main_kpt_preds,
            grid_kpt_pred=grid_kpt_pred,
            kpt_gt=kpt_gt,
            kptness_gt=kptness_gt,
            per_img_threshold=per_img_threshold,
        )
        print("NOISE", std, metrics)

        return metrics["methodA__AVG_ACC"], metrics["GRID_AP"]

    # --------------------------------------------------
    # Compare different noise levels
    # --------------------------------------------------
    acc_low, ap_low = run_with_noise(std=0.2)
    acc_mid, ap_mid = run_with_noise(std=1.0)
    acc_high, ap_high = run_with_noise(std=3.0)

    # Sanity bounds
    for acc, ap in [(acc_low, ap_low), (acc_mid, ap_mid), (acc_high, ap_high)]:
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= ap <= 1.0

    # Monotonic degradation
    assert acc_low > acc_mid > acc_high
    assert ap_low > ap_mid > ap_high


if __name__ == "__main__":
    _test_compute_metrics_perfect_prediction()
    _test_compute_metrics_grid_masking()
    _test_compute_metrics_grid_all_wrong()
    _test_compute_metrics_with_noise()
