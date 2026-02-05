import random
import time
from multiprocessing import Pool, cpu_count, freeze_support
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.signal
from joblib import Parallel, delayed

# Constants
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
EXTENDED_LEADS = LEADS + ["II_short"]
MAX_TIME_SHIFT = 0.2
PERFECT_SCORE = 1e6


def compute_snr(signal_power: float, noise_power: float) -> float:
    if noise_power == 0:
        return PERFECT_SCORE
    elif signal_power == 0:
        return 0
    return min((signal_power / noise_power), PERFECT_SCORE)


def compute_power(label: np.ndarray, prediction: np.ndarray) -> Tuple[float, float]:
    # Sanitize NaNs in prediction for the noise calculation (treat as 0/silence)
    # mirroring the original logic
    pred_clean = prediction.copy()
    pred_clean[~np.isfinite(pred_clean)] = 0
    assert len(label) == len(prediction)

    noise = label - pred_clean
    p_signal = np.sum(label**2)
    p_noise = np.sum(noise**2)
    return p_signal, p_noise


def align_signals_fast(
    label: np.ndarray, pred: np.ndarray, max_shift_samples: int
) -> np.ndarray:
    """
    Optimized alignment:
    1. FFT-based Cross-Correlation for time shift.
    2. Direct mean-difference calculation for vertical shift.
    """
    label = np.asarray(label, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    # 1. Time Alignment
    label_centered = label - np.mean(label)
    pred_centered = pred - np.mean(pred)

    # Fast correlation using FFT
    correlation = scipy.signal.correlate(label_centered, pred_centered, mode="full")

    n_label = len(label)
    n_pred = len(pred)
    lags = scipy.signal.correlation_lags(n_label, n_pred, mode="full")

    # Apply max shift constraint
    valid_mask = (lags >= -max_shift_samples) & (lags <= max_shift_samples)

    # Find max correlation within valid window (Faster: no full array copy)
    max_corr_val = np.max(correlation[valid_mask])

    # Handle ties: find indices in the FULL array that match this max
    candidates = np.flatnonzero(correlation == max_corr_val)

    # CRITICAL SAFETY: Filter out candidates that are technically equal to max
    # but lie outside the valid shift window (e.g. periodic signals)
    candidates = candidates[valid_mask[candidates]]

    if len(candidates) > 0:
        best_idx = min(candidates, key=lambda i: abs(lags[i]))
        time_shift = lags[best_idx]
    else:
        time_shift = 0

    # Construct aligned array with padding
    start_pad = max(time_shift, 0)
    pred_start = max(-time_shift, 0)
    pred_end = min(n_label - time_shift, n_pred)
    end_pad = max(n_label - n_pred - time_shift, 0)

    aligned_pred = np.concatenate(
        (
            np.full(start_pad, np.nan),
            pred[pred_start:pred_end],
            np.full(end_pad, np.nan),
        )
    )

    # 2. Vertical Alignment (Optimized: Direct Mean Difference)
    diff = label - aligned_pred

    # Only calculate shift where we have overlap
    if np.any(np.isfinite(diff)):
        vertical_shift = np.nanmean(diff)
        aligned_pred += vertical_shift

    return aligned_pred


def _process_single_image(args) -> float:
    """Worker function to process one image (12 leads)."""
    pred_dict, gt_dict, fs, do_align = args

    sum_signal = 0.0
    sum_noise = 0.0
    max_shift_samples = int(fs * MAX_TIME_SHIFT)
    for lead, pred in pred_dict.items():
        assert lead in EXTENDED_LEADS
        # Get signals, default to empty/zeros if missing (though they shouldn't be)
        if lead == "II_short":
            ii_signal = gt_dict["II"]
            label = ii_signal[: len(ii_signal) // 4]
        else:
            label = gt_dict[lead]

        # Basic validation to match original strictness, or skip
        if label is None or pred is None:
            continue

        if do_align:
            aligned_pred = align_signals_fast(label, pred, max_shift_samples)
        else:
            aligned_pred = pred
        p_sig, p_noise = compute_power(label, aligned_pred)

        sum_signal += p_sig
        sum_noise += p_noise

    return compute_snr(sum_signal, sum_noise)


def compute_metrics(
    all_pred_signals: Dict[str, Dict[str, np.ndarray]],
    all_gt_signals: Dict[str, Dict[str, np.ndarray]],
    all_fs: Dict[str, float],
    do_align: bool = True,
    num_workers=16,
) -> Tuple[float, Dict[str, float]]:
    """
    Computes the competition metric (Mean SNR in dB) accepting Dictionaries.
    Uses Joblib for parallel processing.

    Args:
        all_pred_signals: Dict mapping sample_id -> {lead_name: np.array}
        all_gt_signals:   Dict mapping sample_id -> {lead_name: np.array}
        all_fs:           Dict mapping sample_id -> sampling_frequency (float)
        do_align:         Whether to perform time/phase alignment (default True)
        num_workers:      Number of parallel jobs. Set to -1 to use all available cores.

    Returns:
        Tuple containing:
        1. Final Score (float): Mean SNR in dB (clipped at -384)
        2. Per-Sample Scores (Dict[str, float]): Dictionary mapping sample_id -> Linear SNR Ratio
    """
    # 1. Type Checks
    if (
        not isinstance(all_pred_signals, dict)
        or not isinstance(all_gt_signals, dict)
        or not isinstance(all_fs, dict)
    ):
        raise TypeError("All inputs (pred, gt, fs) must be dictionaries.")

    # 2. Key Validation: Ensure strict 1:1 mapping across all inputs
    pred_keys = set(all_pred_signals.keys())
    gt_keys = set(all_gt_signals.keys())
    fs_keys = set(all_fs.keys())

    if pred_keys != gt_keys or gt_keys != fs_keys:
        missing = (gt_keys ^ pred_keys) | (gt_keys ^ fs_keys)
        raise ValueError(
            f"Input dictionaries do not share the exact same keys. Mismatch examples: {list(missing)[:5]}"
        )

    # 3. Sort keys for deterministic processing order
    sorted_ids = sorted(list(gt_keys))

    if not sorted_ids:
        return -384.0, {}

    # 4. Prepare Task List
    # Structure: (pred_dict, gt_dict, fs, do_align)
    tasks = [
        (all_pred_signals[sid], all_gt_signals[sid], all_fs[sid], do_align)
        for sid in sorted_ids
    ]

    # 5. Parallel Execution with Joblib
    # Logic to handle num_workers defaulting to max_cpu - 1 if None is passed,
    # or passing -1 directly to joblib to use all cores.
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # Note: We assume _process_single_image is defined in the global scope
    # or imported similarly to how it was used in multiprocessing.Pool
    linear_scores_list = Parallel(n_jobs=num_workers, backend="loky")(
        delayed(_process_single_image)(task) for task in tasks
    )

    # 6. Reconstruct Results
    # Zip the IDs back with their scores
    sample_scores = dict(zip(sorted_ids, linear_scores_list))

    # 7. Final Metric Calculation
    # The metric is 10 * log10( mean( linear_ratios ) )
    all_scores_array = np.array(linear_scores_list)
    mean_snr = np.mean(all_scores_array)

    final_score = -384.0
    if mean_snr > 0:
        final_score = max(10 * np.log10(mean_snr), -384.0)

    return float(final_score), sample_scores


# ==========================================
# Test Data Generator
# ==========================================


def _generate_dummy_test_data(num_samples=500):
    print(f"Generating data for {num_samples} samples...")
    fs_options = [250, 256, 500, 512, 1000, 1025]

    # Structures for Fast Metric (Now Dictionaries)
    fast_gt_dict = {}
    fast_pred_dict = {}
    fast_fs_dict = {}

    # Structures for Slow Metric (DataFrames)
    slow_data_true = {"id": [], "fs": [], "value": []}
    slow_data_pred = {"id": [], "fs": [], "value": []}

    for i in range(num_samples):
        # We use the string representation of 'i' as the sample_id
        sample_id = str(i)

        fs = int(np.random.choice(fs_options))
        fast_fs_dict[sample_id] = fs

        patient_gt = {}
        patient_pred = {}

        for lead in [
            "I",
            "II",
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
        ]:
            # Determine duration
            duration = 10.0 if lead == "II" else 2.5
            n_samples = int(duration * fs)

            # 1. Generate Ground Truth (Sine waves + random noise)
            t = np.linspace(0, duration, n_samples)
            freq = np.random.uniform(0.5, 2.0)  # Heart-rate-ish frequency
            signal = np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.1, n_samples)

            # 2. Generate Prediction (Distorted version of GT)
            # Add Vertical Shift (DC offset)
            dc_offset = np.random.uniform(-2, 2)

            # Add Time Shift (some within 0.2s, some outside to test constraints)
            shift_time = np.random.uniform(-0.3, 0.3)
            shift_samples = int(shift_time * fs)

            # Create prediction array
            pred_signal = (
                np.roll(signal, shift_samples)
                + dc_offset
                + np.random.normal(0, 0.05, n_samples)
            )

            # Store for Fast Metric
            patient_gt[lead] = signal
            patient_pred[lead] = pred_signal

            # Store for Slow Metric
            # ID format: {image_id}_{row_idx}_{lead_name}
            # We match image_id to our sample_id (str(i))
            ids = [f"{sample_id}_{j}_{lead}" for j in range(n_samples)]

            slow_data_true["id"].extend(ids)
            slow_data_true["fs"].extend([fs] * n_samples)
            slow_data_true["value"].extend(signal)

            slow_data_pred["id"].extend(ids)
            slow_data_pred["fs"].extend([fs] * n_samples)
            slow_data_pred["value"].extend(pred_signal)

        fast_gt_dict[sample_id] = patient_gt
        fast_pred_dict[sample_id] = patient_pred

    print("Converting to DataFrames for slow metric (this might take a moment)...")
    df_true = pd.DataFrame(slow_data_true)
    df_pred = pd.DataFrame(slow_data_pred)

    return (fast_pred_dict, fast_gt_dict, fast_fs_dict), (df_true, df_pred)


# ==========================================
# PART 3: Execution & Comparison
# ==========================================

if __name__ == "__main__":

    # Import the original slow metric
    try:
        from ecg.utils.comp_metrics_slow import score as score_slow
    except ImportError:
        print(
            "Error: Could not import 'comp_metrics_slow.py'. Checking local folder fallback..."
        )
        try:
            from comp_metrics_slow import score as score_slow
        except ImportError:
            print("Error: Could not find 'comp_metrics_slow.py'.")
            exit(1)

    freeze_support()

    # 1. Generate Data (Using smaller sample size for quicker local testing if needed)
    (fast_inputs_p, fast_inputs_g, fast_inputs_fs), (df_true, df_pred) = (
        _generate_dummy_test_data(500)
    )
    print(f"Data ready. Total DataFrame rows: {len(df_true)}")
    print("-" * 40)

    # 2. Run Slow Metric
    print("Running ORIGINAL (Slow) Metric...")
    start_time = time.time()
    score_s = score_slow(df_true, df_pred, row_id_column_name="id")
    slow_duration = time.time() - start_time
    print(f"Slow Result: {score_s:.4f}")
    print(f"Slow Time:   {slow_duration:.4f} seconds")
    print("-" * 40)

    # 3. Run Fast Metric
    print("Running OPTIMIZED (Fast) Metric...")
    start_time = time.time()
    # IMPORTANT: do_align must be True to match the slow metric's behavior
    score_f, sample_scores = compute_metrics(
        fast_inputs_p, fast_inputs_g, fast_inputs_fs, do_align=True
    )
    fast_duration = time.time() - start_time
    print(f"Fast Result: {score_f:.4f}")
    print(f"Fast Time:   {fast_duration:.4f} seconds")
    print("-" * 40)

    # 4. Compare
    print("COMPARISON:")

    # Check correctness (tolerance 1e-4 due to float math differences)
    if np.isclose(score_s, score_f, atol=1e-4):
        print("✅ SUCCESS: Scores match!")
    else:
        print(f"❌ FAILURE: Scores diverge! Diff: {abs(score_s - score_f)}")
        print("Note: If failure is small, check do_align=True vs False")

    speedup = slow_duration / fast_duration
    print(f"🚀 Speedup Factor: {speedup:.2f}x")

    # Optional: Print first few per-sample scores to verify dict works
    print("-" * 40)
    print(f"Sample Per-Instance Scores (First 3):")
    for k in list(sample_scores.keys())[:3]:
        print(f"ID {k}: {sample_scores[k]:.4f}")
