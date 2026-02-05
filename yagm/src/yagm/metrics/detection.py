import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_precision_recall_curve_fast(
    matches: np.ndarray, scores: np.ndarray, total_positives: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate precision-recall curve.

    Args:
        matches (np.ndarray): A binary array where 1 indicates a true positive and 0 indicates a false positive.
        scores (np.ndarray): Confidence scores or predicted probabilities for each match.
        total_positives (int): Total number of positive instances (ground truth positives).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Precision, recall, and thresholds for the precision-recall curve.
        Precision is increasing from min_precision to 1.0
        Recall is decreasing from max_recall to 0
        Thresholds is increasing from min_threshold to max_threshold
    """
    matches = matches.astype(bool)
    assert matches.sum() <= total_positives
    if len(matches) == 0:
        return np.array([1]), np.array([0]), np.array([])  # No matches, return default

    # Sort by confidence score in descending order
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_matches = matches[sorted_indices]

    # Identify distinct score values and their corresponding indices
    distinct_value_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct_value_indices, matches.size-1]
    thresholds = sorted_scores[threshold_indices]

    # Cumulative sums of true positives (TP) and false positives (FP)
    true_positives = np.cumsum(sorted_matches)[threshold_indices]
    false_positives = np.cumsum(~sorted_matches)[threshold_indices]

    # Compute precision and recall
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_positives

    # Handle any potential NaN values in precision (e.g., when there are no positive predictions)
    precision = np.nan_to_num(precision, nan=0)

    # Ensure recall decreases and precision is computed for each threshold
    last_index = true_positives.searchsorted(true_positives[-1])
    reverse_slice = slice(last_index, None, -1)

    # Final precision is 1, and final recall is 0 (at the extreme thresholds)
    final_precision = np.r_[precision[reverse_slice], 1]
    final_recall = np.r_[recall[reverse_slice], 0]
    final_thresholds = thresholds[reverse_slice]

    return final_precision, final_recall, final_thresholds


def compute_partial_average_precision_score(
    precision: np.ndarray,
    recall: np.ndarray,
    recall_cutoff=0.0,
    rescale=True,
    method="trapz",
) -> float:
    """
    Compute the Average Precision (AP) score using the precision-recall curve.

    Args:
        precision (np.ndarray): Array of precision values, increasing to 1.0
        recall (np.ndarray): Array of recall values, decreasing to 0.0

    Returns:
        float: The Average Precision (AP) score.
    """
    # Avoid division by zero
    precision = np.nan_to_num(precision, nan=0)

    if not (0 <= recall_cutoff <= 1.0):
        raise ValueError("`recall_cutoff` must be in range[0, 1]")
    assert len(precision) == len(recall)
    if len(recall) < 2:
        assert precision[0] == 1.0 and recall[0] == 0.0
        return 0.0
    cutoff_idx = int((recall >= recall_cutoff).sum())
    if recall_cutoff == 0.0:
        assert cutoff_idx == len(recall)

    if cutoff_idx == 0:
        return 0.0
    elif cutoff_idx == 1:
        # if just one point, interpolate the second point
        precision_interpolate = precision[0] + (precision[1] - precision[0]) * (
            recall[0] - recall_cutoff
        ) / (recall[0] - recall[1])
        # assert precision[0] <= precision_interpolate <= precision[1], f'{recall_cutoff} {precision[:5]} {recall[:5]} {precision_interpolate}'
        # if exactly the same point, return 0
        if precision[0] == precision_interpolate and recall[0] == recall_cutoff:
            return 0.0
        precision = [precision[0], precision_interpolate]
        recall = [recall[0], recall_cutoff]
    else:
        precision = precision[:cutoff_idx]
        recall = recall[:cutoff_idx]

    assert len(precision) == len(recall) >= 2
    if method == "trapz":
        # trapezoidal rule
        pap = -np.trapz(precision, recall)
    elif method == "left_riemann_sum":
        # left Riemann sum
        pap = -np.sum(np.diff(recall) * precision[:-1])
    else:
        raise ValueError
    if rescale:
        pap = pap / (1.0 - recall_cutoff)
    return pap


def compute_det_pr_curve(
    gts, preds, pred_confs, match_func, recall_cutoff=0.0
) -> float:
    """
    Compute Detection Precision-Recall curve + Partial Average Precision score.

    Args:
        gts: list of GT objects
        preds: list of predicted objects
        pred_confs: confident scores for each predicted objects
        match_func: function for bi-partial matching.
            Usually use Greedy/Linear Assigntment matching algorithm.
            This function must has the following signature:
            ```
            example_match_func(gts: List, preds: List, pred_confs: Sequence[float]) -> List[Tuple[int, int]]
            ```
            and return a list of 2 int tuple, which tuple contain the corresponding GT and PRED matched indices.

    Returns:
        Tuple of (precisions, recalls, PAP)
    """
    matches = match_func(gts, preds, pred_confs)
    matches_bin = np.zeros((preds.shape[0],), dtype=bool)
    matches_bin[[e[0] for e in matches]] = True

    precisions, recalls, conf_thresholds = compute_precision_recall_curve_fast(
        matches_bin, pred_confs, len(gts)
    )

    assert len(precisions) == len(recalls) == len(conf_thresholds) + 1
    partial_average_precision = compute_partial_average_precision_score(
        precisions,
        recalls,
        recall_cutoff=recall_cutoff,
        rescale=True,
    )
    return precisions, recalls, partial_average_precision
