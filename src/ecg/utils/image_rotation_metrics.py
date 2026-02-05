import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

EPS = 1e-6


def angle_diff_batch(vectors1, vectors2):
    """
    vector: (N, 2)
    """
    dot_product = np.sum(vectors1 * vectors2, axis=1)
    magnitude1 = np.linalg.norm(vectors1, axis=1)
    magnitude2 = np.linalg.norm(vectors2, axis=1)
    cosine = dot_product / (magnitude1 * magnitude2 + EPS)
    cosine = np.clip(cosine, 0.0, 1.0)
    # print(cosine.tolist())
    angle = np.arccos(cosine)
    angle = np.rad2deg(angle)
    return angle


def compute_metrics(gt_cls, gt_angle, gt_sine_cosine, pred_cls_prob, pred_reg):
    assert (
        len(gt_cls)
        == len(gt_angle)
        == len(gt_sine_cosine)
        == len(pred_cls_prob)
        == len(pred_reg)
    )
    metrics = {}

    pred_cls = np.argmax(pred_cls_prob, axis=-1)
    metrics["accuracy"] = accuracy_score(gt_cls, pred_cls)
    metrics["confusion_matrix"] = confusion_matrix(
        gt_cls, pred_cls, labels=[0, 1, 2, 3]
    )

    # angle differences (MAEs)
    angle_diffs = angle_diff_batch(gt_sine_cosine, pred_reg)
    metrics["angle_mae"] = np.mean(np.abs(angle_diffs))

    # sine/cosine MAE
    metrics["sine_cosine_mae"] = np.mean(np.abs(gt_sine_cosine - pred_reg))

    return metrics
