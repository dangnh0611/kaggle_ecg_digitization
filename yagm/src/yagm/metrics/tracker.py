import re
from typing import Any, Dict
from tabulate import tabulate
import logging
from collections import OrderedDict
import math

logger = logging.getLogger(__name__)


class MetricsTracker:

    def __init__(
        self,
        metrics: Dict[str, str],
        fmt: str = "{metric}/{model_name}",
        keep_top_k: int = -1,
    ):
        # @TODO: support parse metric + model_name from regex
        assert (
            fmt == "{metric}/{model_name}"
        ), "Format `%s` is not supported at this moment!"

        # self.metric2mode = metrics
        self.metric2mode = OrderedDict()
        associate_metrics = {}
        for metric_name, metric_mode in metrics.items():
            if metric_mode in ["min", "max"]:
                self.metric2mode[metric_name] = metric_mode
            else:
                assert (
                    metric_mode in metrics
                ), f"Associated metric `{metric_name}` with mode `{metric_mode}` is invalid, not in metric list {list(metrics.keys())}"
                associate_metrics[metric_name] = metric_mode
        # associate_metrics must go after min-max metrics to ensure correct updating order
        self.metric2mode.update(associate_metrics)
        del metrics

        # assert not any(["train" in name for name in self.metric2mode.keys()
        #                 ]), "Train metric is not supported"
        # assert all([
        #     "val" in name or "test" in name
        #     for name in self.metric2mode.keys()
        # ]), "Metric name must contain `val` or `test`"
        self.fmt = fmt
        self.keep_top_k = keep_top_k
        self.all_best_metrics = {
            metric_name: [] for metric_name in self.metric2mode.keys()
        }
        self.last_best_instance_metrics = {
            metric_name: None for metric_name in self.metric2mode.keys()
        }

    def _find_matched_names(self, names, key):
        # @TODO: make this robust :(
        fmt = (
            self.fmt.replace("{metric}", key)
            .replace("{model_name}", ".")
            .replace("(", "\(")
            .replace(")", "\)")
        )
        return [
            name
            for name in names
            if re.match(fmt, name)
            if not any([name.endswith(k) for k in ["/_best_", "/_primary_"]])
        ]

    def is_equal(self, a, b, excludes=["metadata", "state"]):
        a = {k: v for k, v in a.items() if k not in excludes}
        b = {k: v for k, v in b.items() if k not in excludes}
        return a == b

    def update(
        self,
        cur_metrics: Dict[str, Any],
        cached_metadatas: Dict[str, Dict[str, Any]],
        epoch: int,
        step: int,
    ) -> None:
        for metric_name in self.all_best_metrics.keys():
            metric_mode = self.metric2mode[metric_name]
            if metric_mode not in ["min", "max"]:
                assert metric_mode in self.all_best_metrics
                # update last best model instance
                associated_entry = self.last_best_instance_metrics[metric_mode]
                if associated_entry:
                    self.last_best_instance_metrics[metric_name] = {
                        "mode": metric_mode,
                        "value": associated_entry["state"][
                            f'{metric_name}/{associated_entry["model"]}'
                        ],
                        "epoch": associated_entry["epoch"],
                        "step": associated_entry["step"],
                        "model": associated_entry["model"],
                        "metadata": associated_entry["metadata"],
                        "state": associated_entry["state"],
                    }
                else:
                    continue

                # update global best
                self.all_best_metrics[metric_name] = []
                for associated_entry in self.all_best_metrics[metric_mode]:
                    self.all_best_metrics[metric_name].append(
                        {
                            "mode": metric_mode,
                            "value": associated_entry["state"][
                                f'{metric_name}/{associated_entry["model"]}'
                            ],
                            "epoch": associated_entry["epoch"],
                            "step": associated_entry["step"],
                            "model": associated_entry["model"],
                            "metadata": associated_entry["metadata"],
                            "state": associated_entry["state"],
                        }
                    )
                continue

            cur_matched_metric_names = self._find_matched_names(
                cur_metrics.keys(), metric_name
            )
            if not cur_matched_metric_names:
                # logger.warning(
                #     "Could not found matched metric `%s` while updating metrics tracker: %s",
                #     metric_name,
                #     list(cur_metrics.keys()),
                # )
                continue
            cur_matched_metrics = [
                (name, cur_metrics[name]) for name in cur_matched_metric_names
            ]

            cur_matched_metrics.sort(
                key=lambda x: (
                    math.isnan(x[1]),
                    x[1] if metric_mode == "min" else -x[1],
                )
            )
            cur_best_metric_key, cur_best_metric = cur_matched_metrics[0]
            cur_best_model_name = cur_best_metric_key.replace(f"{metric_name}/", "")
            assert cur_best_model_name not in ["_best_", "_primary_"]
            # update best metrics
            best_metrics = self.all_best_metrics[metric_name]
            metadata = cached_metadatas[cur_best_model_name]
            new_entry = {
                "mode": metric_mode,
                "value": cur_best_metric,
                "epoch": epoch,
                "step": step,
                "model": cur_best_model_name,
                "metadata": metadata,
                "state": cur_metrics,
            }
            old_entries = [e for e in best_metrics if e["step"] == step]
            if old_entries:
                assert len(old_entries) == 1
                old_entry = old_entries[0]
                if not self.is_equal(new_entry, old_entry):
                    logger.warning(
                        "Add new metric entry `%s` with same step %d but different value: old=%f, new=%f",
                        metric_name,
                        step,
                        old_entry["value"],
                        new_entry["value"],
                    )
                    best_metrics.append(new_entry)
                    self.last_best_instance_metrics[metric_name] = new_entry
                else:
                    # no change -> just do nothing
                    old_entry["state"].update(new_entry["state"])
                    continue
            else:
                best_metrics.append(new_entry)
                self.last_best_instance_metrics[metric_name] = new_entry
            # this implementation is more explicit
            # https://stackoverflow.com/questions/1915376/is-pythons-sorted-function-guaranteed-to-be-stable
            best_metrics.sort(
                key=lambda x: (
                    math.isnan(x["value"]),
                    x["value"] if metric_mode == "min" else -x["value"],
                    -x["epoch"],
                    -x["step"],
                )
            )
            # alternative one since python sort() is stable
            # sort by epoch, then by step, and finally by value
            # best_metrics.sort(key = lambda x: x['epoch'], reverse=True)
            # best_metrics.sort(key = lambda x: x['step'], reverse=True)
            # best_metrics.sort(key = lambda x: x['value'], reverse = (metric_mode == 'max'))

            if self.keep_top_k > 0:
                self.all_best_metrics[metric_name] = best_metrics[: self.keep_top_k]

    def find_top_k(self, metric_name=None):
        raise NotImplementedError

    @property
    def best_metrics(self):
        return {k: (v[0] if v else None) for k, v in self.all_best_metrics.items()}

    def repr_table(self, top_k=-1, fmt="rounded_grid"):
        headers = ["rank", "metric", "value", "mode", "model", "epoch", "step"]
        rows = []
        all_metric_names = sorted(list(self.all_best_metrics.keys()))
        for metric_name in all_metric_names:
            best_metrics = self.all_best_metrics[metric_name]
            for i, entry in enumerate(best_metrics):
                new_row = [
                    i,
                    metric_name,
                    entry["value"],
                    entry["mode"],
                    entry["model"],
                    entry["epoch"],
                    entry["step"],
                ]
                rows.append(new_row)
                if top_k > 0 and i >= top_k:
                    break
        table = tabulate(rows, headers=headers, tablefmt=fmt)
        return table
