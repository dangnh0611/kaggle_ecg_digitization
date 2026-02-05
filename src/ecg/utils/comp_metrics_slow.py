from typing import Tuple

import numpy as np
import pandas as pd

import scipy.optimize
import scipy.signal


LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
MAX_TIME_SHIFT = 0.2
PERFECT_SCORE = 1e6


class ParticipantVisibleError(Exception):
    pass


def compute_power(label: np.ndarray, prediction: np.ndarray) -> Tuple[float, float]:
    if label.ndim != 1 or prediction.ndim != 1:
        raise ParticipantVisibleError('Inputs must be 1-dimensional arrays.')
    finite_mask = np.isfinite(prediction)
    if not np.any(finite_mask):
        raise ParticipantVisibleError("The 'prediction' array contains no finite values (all NaN or inf).")

    prediction[~np.isfinite(prediction)] = 0
    noise = label - prediction
    p_signal = np.sum(label**2)
    p_noise = np.sum(noise**2)
    return p_signal, p_noise


def compute_snr(signal: float, noise: float) -> float:
    if noise == 0:
        # Perfect reconstruction
        snr = PERFECT_SCORE
    elif signal == 0:
        snr = 0
    else:
        snr = min((signal / noise), PERFECT_SCORE)
    return snr


def align_signals(label: np.ndarray, pred: np.ndarray, max_shift: float = float('inf')) -> np.ndarray:
    if np.any(~np.isfinite(label)):
        raise ParticipantVisibleError('values in label should all be finite')
    if np.sum(np.isfinite(pred)) == 0:
        raise ParticipantVisibleError('prediction can not all be infinite')

    # Initialize the reference and digitized signals.
    label_arr = np.asarray(label, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)

    label_mean = np.mean(label_arr)
    pred_mean = np.mean(pred_arr)

    label_arr_centered = label_arr - label_mean
    pred_arr_centered = pred_arr - pred_mean

    # Compute the correlation between the reference and digitized signals and locate the maximum correlation.
    correlation = scipy.signal.correlate(label_arr_centered, pred_arr_centered, mode='full')

    n_label = np.size(label_arr)
    n_pred = np.size(pred_arr)

    lags = scipy.signal.correlation_lags(n_label, n_pred, mode='full')
    valid_lags_mask = (lags >= -max_shift) & (lags <= max_shift)

    max_correlation = np.nanmax(correlation[valid_lags_mask])
    all_max_indices = np.flatnonzero(correlation == max_correlation)
    best_idx = min(all_max_indices, key=lambda i: abs(lags[i]))
    time_shift = lags[best_idx]
    start_padding_len = max(time_shift, 0)
    pred_slice_start = max(-time_shift, 0)
    pred_slice_end = min(n_label - time_shift, n_pred)
    end_padding_len = max(n_label - n_pred - time_shift, 0)
    aligned_pred = np.concatenate((np.full(start_padding_len, np.nan), pred_arr[pred_slice_start:pred_slice_end], np.full(end_padding_len, np.nan)))

    def objective_func(v_shift):
        return np.nansum((label_arr - (aligned_pred - v_shift)) ** 2)

    if np.any(np.isfinite(label_arr) & np.isfinite(aligned_pred)):
        results = scipy.optimize.minimize_scalar(objective_func, method='Brent')
        vertical_shift = results.x
        aligned_pred -= vertical_shift
    return aligned_pred


def _calculate_image_score(group: pd.DataFrame) -> float:
    """Helper function to calculate the total SNR score for a single image group."""

    unique_fs_values = group['fs'].unique()
    if len(unique_fs_values) != 1:
        raise ParticipantVisibleError('Sampling frequency should be consistent across each ecg')
    sampling_frequency = unique_fs_values[0]
    if sampling_frequency != int(len(group[group['lead'] == 'II']) / 10):
        raise ParticipantVisibleError('The sequence_length should be sampling frequency * 10s')
    sum_signal = 0
    sum_noise = 0
    for lead in LEADS:
        sub = group[group['lead'] == lead]
        label = sub['value_true'].values
        pred = sub['value_pred'].values

        aligned_pred = align_signals(label, pred, int(sampling_frequency * MAX_TIME_SHIFT))
        p_signal, p_noise = compute_power(label, aligned_pred)
        sum_signal += p_signal
        sum_noise += p_noise
    return compute_snr(sum_signal, sum_noise)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Compute the mean Signal-to-Noise Ratio (SNR) across multiple ECG leads and images for the PhysioNet 2025 competition.
    The final score is the average of the sum of SNRs over different lines, averaged over all unique images.
    Args:
        solution: DataFrame with ground truth values. Expected columns: 'id' and one for each lead.
        submission: DataFrame with predicted values. Expected columns: 'id' and one for each lead.
        row_id_column_name: The name of the unique identifier column, typically 'id'.
    Returns:
        The final competition score.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> row_id_column_name = "id"
    >>> solution = pd.DataFrame({'id': ['343_0_I', '343_1_I', '343_2_I', '343_0_III', '343_1_III','343_2_III','343_0_aVR', '343_1_aVR','343_2_aVR',\
    '343_0_aVL', '343_1_aVL', '343_2_aVL', '343_0_aVF', '343_1_aVF','343_2_aVF','343_0_V1', '343_1_V1', '343_2_V1','343_0_V2', '343_1_V2','343_2_V2',\
    '343_0_V3', '343_1_V3', '343_2_V3','343_0_V4', '343_1_V4', '343_2_V4', '343_0_V5', '343_1_V5','343_2_V5','343_0_V6', '343_1_V6','343_2_V6',\
    '343_0_II', '343_1_II','343_2_II', '343_3_II', '343_4_II', '343_5_II','343_6_II', '343_7_II','343_8_II','343_9_II','343_10_II','343_11_II'],\
    'fs': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\
    'value':[0.1,0.3,0.4,0.6,0.6,0.4,0.2,0.3,0.4,0.5,0.2,0.7,0.2,0.3,0.4,0.8,0.6,0.7, 0.2,0.3,-0.1,0.5,0.6,0.7,0.2,0.9,0.4,0.5,0.6,0.7,0.1,0.3,0.4,\
    0.6,0.6,0.4,0.2,0.3,0.4,0.5,0.2,0.7,0.2,0.3,0.4]})
    >>> submission = solution.copy()
    >>> round(score(solution, submission, row_id_column_name), 4)
    60.0
    >>> submission.loc[0, 'value'] = 0.9 # Introduce some noise
    >>> round(score(solution, submission, row_id_column_name), 4)
    13.6291
    >>> submission.loc[4, 'value'] = 0.3 # Introduce some noise
    >>> round(score(solution, submission, row_id_column_name), 4)
    13.0576

    >>> solution = pd.DataFrame({'id': ['343_0_I', '343_1_I', '343_2_I', '343_0_III', '343_1_III','343_2_III','343_0_aVR', '343_1_aVR','343_2_aVR',\
    '343_0_aVL', '343_1_aVL', '343_2_aVL', '343_0_aVF', '343_1_aVF','343_2_aVF','343_0_V1', '343_1_V1', '343_2_V1','343_0_V2', '343_1_V2','343_2_V2',\
    '343_0_V3', '343_1_V3', '343_2_V3','343_0_V4', '343_1_V4', '343_2_V4', '343_0_V5', '343_1_V5','343_2_V5','343_0_V6', '343_1_V6','343_2_V6',\
    '343_0_II', '343_1_II','343_2_II', '343_3_II', '343_4_II', '343_5_II','343_6_II', '343_7_II','343_8_II','343_9_II','343_10_II','343_11_II'],\
    'fs': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\
    'value':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]})
    >>> round(score(solution, submission, row_id_column_name), 4)
    -384
    >>> submission = solution.copy()
    >>> round(score(solution, submission, row_id_column_name), 4)
    60.0

    >>> # test alignment
    >>> label = np.array([0, 1, 2, 1, 0])
    >>> pred = np.array([0, 1, 2, 1, 0])
    >>> aligned = align_signals(label, pred)
    >>> expected_array = np.array([0, 1, 2, 1, 0])
    >>> np.allclose(aligned, expected_array, equal_nan=True)
    True

    >>> # Test 2: Vertical shift (DC offset) should be removed
    >>> label = np.array([0, 1, 2, 1, 0])
    >>> pred = np.array([10, 11, 12, 11, 10])
    >>> aligned = align_signals(label, pred)
    >>> expected_array = np.array([0, 1, 2, 1, 0])
    >>> np.allclose(aligned, expected_array, equal_nan=True)
    True

    >>> # Test 3: Time shift should be corrected
    >>> label = np.array([0, 0, 1, 2, 1, 0., 0.])
    >>> pred = np.array([1, 2, 1, 0, 0, 0, 0])
    >>> aligned = align_signals(label, pred)
    >>> expected_array = np.array([np.nan, np.nan, 1, 2, 1, 0, 0])
    >>> np.allclose(aligned, expected_array, equal_nan=True)
    True
    
    >>> # Test 4: max_shift constraint prevents optimal alignment
    >>> label = np.array([0, 0, 0, 0, 1, 2, 1]) # Peak is far
    >>> pred = np.array([1, 2, 1, 0, 0, 0, 0])
    >>> aligned = align_signals(label, pred, max_shift=10)
    >>> expected_array = np.array([ np.nan, np.nan, np.nan, np.nan, 1, 2, 1])
    >>> np.allclose(aligned, expected_array, equal_nan=True)
    True

    """
    for df in [solution, submission]:
        if row_id_column_name not in df.columns:
            raise ParticipantVisibleError(f"'{row_id_column_name}' column not found in DataFrame.")
        if df['value'].isna().any():
            raise ParticipantVisibleError('NaN exists in solution/submission')
        if not np.isfinite(df['value']).all():
            raise ParticipantVisibleError('Infinity exists in solution/submission')

    submission = submission[['id', 'value']]
    merged_df = pd.merge(solution, submission, on=row_id_column_name, suffixes=('_true', '_pred'))
    merged_df['image_id'] = merged_df[row_id_column_name].str.split('_').str[0]
    merged_df['row_id'] = merged_df[row_id_column_name].str.split('_').str[1].astype('int64')
    merged_df['lead'] = merged_df[row_id_column_name].str.split('_').str[2]
    merged_df.sort_values(by=['image_id', 'row_id', 'lead'], inplace=True)
    image_scores = merged_df.groupby('image_id').apply(_calculate_image_score, include_groups=False)
    return max(float(10 * np.log10(image_scores.mean())), -384)