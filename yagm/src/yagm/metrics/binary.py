import numpy as np


def compute_fbeta_score(
    precision: np.ndarray, recall: np.ndarray, beta: float
) -> np.ndarray:
    """
    Calculate the F-beta score for a range of precision and recall values.

    Args:
        precision (np.ndarray): An array of precision values for different thresholds.
        recall (np.ndarray): An array of recall values for different thresholds.
        beta (float): The beta parameter to control the weight of recall (e.g., beta = 1 is F1 score, beta > 1 emphasizes recall).

    Returns:
        np.ndarray: An array of F-beta scores for each threshold.
    """
    # Ensure that precision and recall are arrays with the same length
    if len(precision) != len(recall):
        raise ValueError("Precision and recall arrays must have the same length.")

    # Calculate the F-beta score for each threshold
    beta_squared = beta**2
    f_beta = (
        (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
    )
    np.nan_to_num(f_beta, copy=False, nan=0.0)
    return f_beta
