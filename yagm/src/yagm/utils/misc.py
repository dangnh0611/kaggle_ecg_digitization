import contextlib
import logging
import math
import os
import pprint
import shutil
import subprocess
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import polars as pl
import torch
from matplotlib import pyplot as plt
from tabulate import tabulate
from torch import nn

logger = logging.getLogger(__name__)


def calculate_steps_per_epoch(
    num_samples: int,
    batch_size: int,
    drop_last: bool = True,
    accumulate_grad_batches: int = 1,
) -> int:
    """Estimated stepping batches for the complete training."""
    num_batches = num_samples / batch_size
    num_batches = int(num_batches) if drop_last else math.ceil(num_batches)
    # This may not be accurate with small error based on how grad_accum implemented.
    # @TODO - determine if int() or math.ceil()
    num_steps = math.ceil(num_batches / accumulate_grad_batches)
    return num_steps


def load_state_dict(model: nn.Module, state_dict: dict, strict: bool = True):
    """Same as torch but not raise error in case of imcompatible shape."""
    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        try:
            model_sd = model.state_dict()
            all_keys = list(set(list(model_sd.keys()) + list(state_dict.keys())))
            share_sd = {}
            redundant_keys = []  # not in model but found in state dict
            missing_keys = []  # in model but not found in state dict
            shape_mismatches = []  # imcompatible shape
            for k in all_keys:
                if k in model_sd and k in state_dict:
                    if model_sd[k].shape == state_dict[k].shape:
                        share_sd[k] = state_dict[k]
                    else:
                        shape_mismatches.append(
                            (k, model_sd[k].shape, state_dict[k].shape)
                        )
                elif k in model_sd and k not in state_dict:
                    missing_keys.append(k)
                elif k not in model_sd and k in state_dict:
                    redundant_keys.append(k)
            if redundant_keys or missing_keys or shape_mismatches:
                logger.warning(
                    "Loading state dict with MISSING=%s\nREDUNDANT=%s\nSHAPE MISMATCHES:%s",
                    missing_keys,
                    redundant_keys,
                    shape_mismatches,
                )
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            # Same keys, but differ in shape
            logger.warning(
                "Error while loading state dict:\n%s\nAttempt load shared state dict only..",
                e,
            )
            model.load_state_dict(share_sd, strict=False)
    return model


def dict_as_table(
    dictionary, headers=["key", "value"], sort_by="key", format="rounded_grid"
):
    table = [[str(k), pprint.pformat(v)] for k, v in dictionary.items()]
    if sort_by is None:
        pass
    elif sort_by == "key":
        table.sort(key=lambda x: x[0])
    elif sort_by == "value":
        table.sort(key=lambda x: x[1])
    else:
        # sort_by is a function
        table.sort(key=sort_by)
    return tabulate(table, headers=headers, tablefmt=format)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        return self.sum / self.count

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n


def get_subplots(n, nrows=None, ncols=None, figsize=None):
    if nrows:
        ncols = math.ceil(n / nrows)
    elif ncols:
        nrows = math.ceil(n / ncols)
    else:
        raise ValueError

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.tight_layout()
    return fig, axes.flat


@contextlib.contextmanager
def local_numpy_temporary_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_xlsx_copiable_metrics(
    best_metrics,
    metric_names=["val/loss"],
    value_round=6,
    time_by="step",
    epoch_round=1,
):
    assert time_by in ["step", "epoch"]
    cells = []
    for name in metric_names:
        if name in best_metrics:
            best_entry = best_metrics[name]
            if best_entry:
                best_entry = best_entry[0]
                cell = f'{round(best_entry["value"], value_round)} ({best_entry["model"]} {round(best_entry["epoch"], epoch_round) if time_by == "epoch" else best_entry["step"]})'
            else:
                cell = "N/A"
        else:
            cell = "N/A"
        cells.append(cell)
    header_str = "	".join(metric_names)
    cell_str = "	".join(cells)
    ret_str = header_str + "\n" + cell_str
    return ret_str


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def save_img_to_file(save_path, img, backend="cv2"):
    file_ext = os.path.basename(save_path).split(".")[-1]
    if backend == "cv2":
        if img.dtype == np.uint16:
            # https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
            assert file_ext in ["png", "jp2", "tiff", "tif"]
            cv2.imwrite(save_path, img)
        elif img.dtype == np.uint8:
            cv2.imwrite(save_path, img)
        else:
            raise ValueError("`cv2` backend only support uint8 or uint16 images.")
    elif backend == "np":
        assert file_ext == "npy"
        np.save(save_path, img)
    else:
        raise ValueError(f"Unsupported backend `{backend}`.")


def load_img_from_file(img_path, backend="cv2"):
    if backend == "cv2":
        return cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    elif backend == "np":
        return np.load(img_path)
    else:
        raise ValueError()


def rm_and_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=False)


def make_symlink(src, dst):
    abs_src = os.path.abspath(src)
    abs_dst = os.path.abspath(dst)
    assert os.path.exists(abs_src)
    os.symlink(abs_src, abs_dst)


def pandas_to_polars_dataframe(df, verbose=False):
    """Easier convert Pandas DataFrame to Polars DataFrame with Object Dtype"""
    primitive_cols = []
    str_cols = []
    object_cols = []
    for col_name, col_dtype in zip(df.columns, df.dtypes):
        if df.at[0, col_name] is not None:
            repr_element = df.at[0, col_name]
        else:
            values = df[~df[col_name].isna()].reset_index()[col_name]
            if len(values):
                repr_element = values[0]
            else:
                repr_element = None
                logger.warn("A column %s contains all NULL values", col_name)
        if verbose:
            logger.info("col_name=%s, dtype=%s", col_name, type(repr_element))
        if col_dtype != np.dtypes.ObjectDType:
            primitive_cols.append(col_name)
            continue
        if isinstance(repr_element, str):
            str_cols.append(col_name)
        elif isinstance(repr_element, np.ndarray) or isinstance(repr_element, list):
            object_cols.append(col_name)
        elif repr_element is None:
            object_cols.append(col_name)
        else:
            object_cols.append(col_name)
    ret_df = pl.from_pandas(df[primitive_cols])
    ret_df = ret_df.with_columns(
        **{col_name: pl.Series(df[col_name], dtype=pl.String) for col_name in str_cols},
        **{
            col_name: pl.Series(df[col_name].to_list(), dtype=pl.Object)
            for col_name in object_cols
        },
    ).select(pl.col(df.columns))
    return ret_df


def create_metrics_summary(
    log_metrics: List[str],
    all_fold_best_metrics: Dict[int, Dict[str, List[Dict[str, Any]]]],
    all_oof_best_metrics: List[Dict[str, Any]] | None = None,
    topk: int = 1,
    csv_path: str | None = None,
    include_fold_mean_std=True,
) -> None:
    """
    Log the metrics summary across all folds + OOF

    Args:
        log_metrics: metric names to be logged
        all_fold_best_metrics: dictionary mapping fold index to a dictionary of fold's best metrics.
            Example:
                {
                    1: {'val/loss': [{'epoch': 20, 'step': 400, 'value': 0.01}]}
                }
        all_oof_best_metrics: each element in this list is a dictionary of OOF metrics (ordered by rank)
        topk: limit the number of entries to be logged/displayed
        csv_path: if not None, save the results table as csv
    """
    all_fold_idxs = sorted(list(all_fold_best_metrics.keys()))
    # fold_idx, topk, metric_name, value, model, epoch, step
    headers = ["rank", "fold", *log_metrics]
    all_rows = []
    # loop through all metrics which need to be logged
    best_values_per_metrics = {}
    for fold_idx in all_fold_idxs:
        fold_best_metrics = all_fold_best_metrics[fold_idx]
        for _top in range(topk):
            row = [_top + 1, fold_idx]
            for metric in log_metrics:
                try:
                    e = fold_best_metrics[metric][_top]
                    value, model, epoch, step = (
                        e["value"],
                        e["model"],
                        e["epoch"],
                        e["step"],
                    )
                    best_values_per_metrics.setdefault(metric, []).append(value)
                    # epoch = int(epoch) if epoch == int(epoch) else epoch
                    # row.append(f"{value:.6f} ({model} {epoch})")
                    step = f"{round(step / 1000, 1)}K"
                    row.append(f"{value:.6f} ({model} {step})")
                except (KeyError, IndexError):
                    row.append("NA")
            assert len(row) == len(headers)
            all_rows.append(row)
    if all_oof_best_metrics is not None:
        for _top, oof_metrics in enumerate(all_oof_best_metrics):
            if _top >= topk:
                break
            row = [_top + 1, "OOF"]
            for metric in log_metrics:
                try:
                    value = oof_metrics[metric]
                    if not isinstance(value, str):
                        value = f"{value:.6f}"
                    if _top == 0 and include_fold_mean_std:
                        metric_mean = float(np.nanmean(best_values_per_metrics[metric]))
                        metric_std = float(np.nanstd(best_values_per_metrics[metric]))
                        logger.info(
                            "FOLD METRICS OF %s with mean=%f std=%f values=%s",
                            metric,
                            metric_mean,
                            metric_std,
                            best_values_per_metrics[metric],
                        )
                        value = f"{value} ({metric_mean:.6f} +- {metric_std:.6f})"
                    row.append(value)
                except KeyError:
                    row.append("NA")
            assert len(row) == len(headers)
            all_rows.append(row)

    # print table
    table_str = tabulate(all_rows, headers=headers, tablefmt="rounded_grid")
    logger.info("METRICS SUMMARY FOR ALL FOLDS + OOF:\n%s\n", table_str)

    metrics_df = pd.DataFrame(
        {
            headers[col_idx]: [
                all_rows[row_idx][col_idx] for row_idx in range(len(all_rows))
            ]
            for col_idx in range(len(headers))
        }
    )
    # log to csv file
    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        logger.info("Logging metrics summary to %s", csv_path)
        metrics_df.to_csv(csv_path, index=False)

    return metrics_df


def get_last_git_logs(count=3):
    try:
        # Run 'git log -n <count>' to get the latest <count> commit logs
        logs = subprocess.check_output(
            ["git", "log", f"-n{count}", "--pretty=format:%H %s"]
        ).decode("utf-8")
        commit_logs = logs.splitlines()  # Split the logs into individual commits
        return commit_logs
    except subprocess.CalledProcessError as e:
        logger.warning("Could not retrieve Git commit logs.")
        return []


@contextlib.contextmanager
def torch_set_num_threads_context(n: int):
    ori_n = torch.get_num_threads()
    torch.set_num_threads(n)
    yield
    torch.set_num_threads(ori_n)


def nan_fill(arr, value):
    num_nan = float(np.isnan(arr).sum())
    if num_nan > 0:
        logger.warning(
            "Detect %d/%d nan predictions, fill to %f", num_nan, arr.size, value
        )
        return np.nan_to_num(arr, nan=value)
    return arr
