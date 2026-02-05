import copy
import logging
import os
import traceback
from importlib.util import find_spec
from typing import Any, Callable, Dict, List, Sequence, Tuple

import hydra
import lightning as L
import numpy as np
import rich.syntax
import rich.tree
import torch
from lightning.pytorch.callbacks import LearningRateFinder, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import Logger as LLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from yagm.data.base_data_module import BaseDataModule
from yagm.tasks.base_task import BaseTask
from yagm.utils import misc as misc_utils

logger = logging.getLogger(__name__)


class VerboseLearningRateFinder(LearningRateFinder):
    def __init__(self, save_dir, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)
            # plot
            fig = self.optimal_lr.plot()
            plt.savefig(
                os.path.join(self.save_dir, f"lr_finder_ep={trainer.current_epoch}.png")
            )
            plt.close()
            suggested_lr = self.optimal_lr.suggestion()
            logger.info(
                "Best learning rate suggestion %f at epoch %d",
                suggested_lr,
                trainer.current_epoch,
            )


class CustomTQDMProgressBar(TQDMProgressBar):
    BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        # print new lines so that next call to print() or log()
        # won't start on the same line as progress bar
        # -> just look more clean :)
        print("\n\n")

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        return super().on_train_epoch_end(trainer, pl_module)

    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


class CustomModelCheckpoint(ModelCheckpoint):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_registry(self, registry: Dict[int, Dict[str, Any]]):
        """
        This must be called right after constructor self.__init__().
        This will inplace update the provided dictionary `registry`
        """
        registry[id(self)] = self.best_k_models
        self._registry = registry
        logger.info("ModelCheckpoint Registry: %s", list(registry.keys()))

    def _should_remove_checkpoint(
        self, trainer: L.Trainer, previous: str, current: str
    ):
        for _cb_id, best_k_models in self._registry.items():
            if previous in best_k_models:
                # this checkpoint is one of the topk best for another metrics
                # keep this, no deletion
                return False

        return super()._should_remove_checkpoint(trainer, previous, current)

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        # If file existed before, no overwrite
        # This save time in case of large model, frequently saving or
        # multiple ModelCheckpoint callbacks with multiple metrics
        if os.path.isfile(filepath):
            if len(self._registry) < 2 and filepath != self.last_model_path:
                logger.warning(
                    "ModelCheckpoint: last checkpoint %s already exists, which is unusual "
                    "except in case of multiple ModelCheckpoint Callbacks were provided.",
                    filepath,
                )
            logger.debug(
                "ModelCheckpoint: checkpoint path %s exist, no overwrite!", filepath
            )
            return
        super()._save_checkpoint(trainer, filepath)

    def _update_best_and_save(self, current, trainer, monitor_candidates) -> None:
        super()._update_best_and_save(current, trainer, monitor_candidates)
        best_model_path = self.best_model_path
        best_score = self.best_model_score
        best_symlink_path = os.path.join(
            self.dirpath, f"best_{self.monitor.replace('/', '_')}.ckpt"
        )
        self._link_checkpoint(trainer, best_model_path, best_symlink_path)


class ValidationScheduler(L.Callback):
    """
    This callback helps to change validation interval adaptively (scheduled) by epochs.
    Author: dangnh0611@gmail.com
    Currently support epoch based, that is, exec this sheduler after each epoch end only.
    Future version will include support step based, with small added overhead
    """

    def __init__(
        self,
        milestones,
        val_check_intervals=None,
        check_val_every_n_epochs=None,
        milestone_unit="epoch",
    ):
        assert len(milestones) < 2 or all(
            [
                after - before > 0
                for before, after in zip(milestones[:-1], milestones[1:])
            ]
        )
        self.milestones = milestones

        if val_check_intervals is not None:
            assert len(self.milestones) == len(val_check_intervals)
            self.val_check_intervals = val_check_intervals
        else:
            self.val_check_intervals = [None] * len(self.milestones)

        if check_val_every_n_epochs is not None:
            assert len(self.milestones) == len(check_val_every_n_epochs)
            self.check_val_every_n_epochs = check_val_every_n_epochs
        else:
            self.check_val_every_n_epochs = [None] * len(self.milestones)
        assert (
            len(self.milestones)
            == len(self.val_check_intervals)
            == len(self.check_val_every_n_epochs)
        )
        self.cur_milestone = None
        self.milestone_unit = milestone_unit

    def get_new_config(self, epoch, steps_per_epoch):
        # get current milestone
        for i, (milestone, val_check_interval, check_val_every_n_epoch) in enumerate(
            zip(
                self.milestones, self.val_check_intervals, self.check_val_every_n_epochs
            )
        ):
            if self.milestone_unit == "epoch":
                pass
            elif self.milestone_unit == "step":
                milestone = int(milestone / steps_per_epoch)
            else:
                raise ValueError
            if epoch >= milestone and self.cur_milestone != milestone:
                if i + 1 < len(self.milestones):
                    next_milestone = self.milestones[i + 1]
                    assert epoch < next_milestone
                self.cur_milestone = milestone
                logger.info(
                    "ValidationScheduler Callback: milestone %d encountered with epoch=%d.",
                    self.cur_milestone,
                    epoch,
                )
                return milestone, val_check_interval, check_val_every_n_epoch
        return None, None, None

    def _set_val_check_interval(self, trainer, value, accumulate_grad_batches):
        # ref: https://github.com/Lightning-AI/pytorch-lightning/blob/a944e7744e57a5a2c13f3c73b9735edf2f71e329/src/lightning/pytorch/loops/fit_loop.py#L287
        if isinstance(value, int):
            trainer.val_check_batch = value * accumulate_grad_batches
        else:
            trainer.val_check_batch = max(1, int(trainer.fit_loop.max_batches * value))

        if (
            trainer.val_check_batch > trainer.fit_loop.max_batches
            and trainer.check_val_every_n_epoch is not None
        ):
            raise ValueError(
                f" `val_check_interval` ({trainer.val_check_interval}) must be less than or equal"
                f" to the number of the training batches ({self.max_batches})."
                " If you want to disable validation set `limit_val_batches` to 0.0 instead."
                " If you want to validate based on the total training batches, set `check_val_every_n_epoch=None`."
            )

    def _set_check_val_every_n_epoch(self, trainer, value):
        trainer.check_val_every_n_epoch = value

    def on_train_epoch_end(self, trainer, pl_module: BaseTask):
        # next_epoch will equal to 1 after the very first epoch
        next_epoch = trainer.current_epoch
        milestone, val_check_interval, check_val_every_n_epoch = self.get_new_config(
            next_epoch, steps_per_epoch=pl_module.steps_per_epoch
        )
        if val_check_interval or check_val_every_n_epoch:
            logger.info(
                "ValidationScheduler Callback: end of epoch=%d, milestone=%f, set val_check_interval=%s check_val_every_n_epoch=%s",
                trainer.current_epoch,
                milestone,
                val_check_interval,
                check_val_every_n_epoch,
            )
        if val_check_interval is not None:
            self._set_val_check_interval(
                trainer, val_check_interval, trainer.accumulate_grad_batches
            )
        if check_val_every_n_epoch is not None:
            self._set_check_val_every_n_epoch(trainer, check_val_every_n_epoch)


class CudaPerformanceBenchmark(L.Callback):
    """A callback that measures the median execution time between the start and end of a batch."""

    def __init__(self, train_interval=None, val_interval=None):
        self.train_interval = train_interval
        self.val_interval = val_interval
        self.train_start = torch.cuda.Event(enable_timing=True)
        self.train_end = torch.cuda.Event(enable_timing=True)
        self.val_start = torch.cuda.Event(enable_timing=True)
        self.val_end = torch.cuda.Event(enable_timing=True)
        self.val_global_step = 0
        self.train_iter_times = []
        self.val_iter_times = []

    def median_time(self):
        return np.median(self.train_iter_times)

    def log_current_stats(self, stage="train"):
        if stage == "train":
            logger.info(
                "[PerfBenchCB] Training speed (%d observation): min=%.1f mean=%.1f median=%.1f std=%.1f max=%.1f ms/iter",
                len(self.train_iter_times),
                np.min(self.train_iter_times),
                np.mean(self.train_iter_times),
                np.median(self.train_iter_times),
                np.std(self.train_iter_times),
                np.max(self.train_iter_times),
            )
        elif stage == "val":
            logger.info(
                "[PerfBenchCB] Validation speed (%d observations): min=%.1f mean=%.1f median=%.1f std=%.1f max=%.1f ms/iter",
                len(self.val_iter_times),
                np.min(self.val_iter_times),
                np.mean(self.val_iter_times),
                np.median(self.val_iter_times),
                np.std(self.val_iter_times),
                np.max(self.val_iter_times),
            )
        else:
            raise ValueError

    def on_train_batch_start(self, trainer, *args, **kwargs):
        self.train_start.record()

    def on_train_batch_end(self, trainer, *args, **kwargs):
        self.train_end.record()
        torch.cuda.synchronize()
        elapsed_time = self.train_start.elapsed_time(self.train_end)
        # Exclude the first iteration to let the model warm up
        if trainer.global_step <= 1:
            logger.info(
                "[PerfBenchCB] First training step take %f ms",
                elapsed_time,
            )
        else:
            self.train_iter_times.append(elapsed_time)

        if (
            self.train_interval is not None
            and trainer.global_step % self.train_interval == 0
        ):
            self.log_current_stats("train")

    def on_validation_batch_start(self, trainer, *args, **kwargs):
        self.val_start.record()

    def on_validation_batch_end(self, trainer, *args, **kwargs):
        self.val_global_step += 1
        self.val_end.record()
        torch.cuda.synchronize()
        elapsed_time = self.val_start.elapsed_time(self.val_end)
        # Exclude the first iteration to let the model warm up
        if self.val_global_step <= 1:
            logger.info(
                "[PerfBenchCB] First validation step take %f ms",
                elapsed_time,
            )
        else:
            self.val_iter_times.append(elapsed_time)

        if (
            self.val_interval is not None
            and self.val_global_step % self.val_interval == 0
        ):
            self.log_current_stats("val")

    def on_validation_end(self, trainer, pl_module) -> None:
        """Called when the validation loop begins."""
        self.log_current_stats("val")
        return super().on_validation_end(trainer, pl_module)

    def on_validation_start(self, trainer, pl_module):
        self.log_current_stats("train")
        return super().on_validation_start(trainer, pl_module)


def build_callbacks(cfg: DictConfig) -> List[L.Callback]:
    """Instantiates callbacks from config.

    Note that ModelCheckpoint support tracking by multiple metrics, specified by
    `cfg.callbacks.model_checkpoint.metrics`
    """
    callbacks: List[L.Callback] = []

    if not hasattr(cfg, "callbacks"):
        logger.warning("No callback configs found! Skipping loading...")
        return callbacks

    if not isinstance(cfg.callbacks, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    model_checkpoint_registry = {}
    for cb_name, cb_conf in cfg.callbacks.items():
        if cb_name == "model_checkpoint":
            if cb_conf.metrics is None:
                metric_names = [None]
            elif isinstance(cb_conf.metrics, str):
                metric_names = [cb_conf.metrics]
            else:
                metric_names = cb_conf.metrics
            
            # @TODO - fix this
            # if cb_conf.every_n_train_steps is not None:
            #     # every_n_train_steps is not work when `monitor` is not None
            #     # and the metric is not computed yet
            #     # explicit add additional one
            #     if None not in metric_names:
            #         metric_names.append(None)

            for metric_name in metric_names:
                new_cb_conf = copy.deepcopy(cb_conf)
                del new_cb_conf.metrics
                if metric_name is not None:
                    save_path = "ep={epoch}_step={step}_" + "_".join(
                        [
                            f"{_metric_name.replace('/', '_')}={{{_metric_name}:.6f}}"
                            for _metric_name in metric_names if _metric_name is not None
                        ]
                    )
                    OmegaConf.update(
                        new_cb_conf, "monitor", metric_name, force_add=True
                    )
                    OmegaConf.update(
                        new_cb_conf,
                        "mode",
                        cfg.task.metrics[metric_name],
                        force_add=True,
                    )
                else:
                    save_path = "ep={epoch}_step={step}"
                    OmegaConf.update(new_cb_conf, "monitor", None, force_add=True)
                    OmegaConf.update(new_cb_conf, "mode", "min", force_add=True)
                OmegaConf.update(
                    new_cb_conf,
                    "filename",
                    save_path,
                    force_add=True,
                )
                new_cb: CustomModelCheckpoint = hydra.utils.instantiate(new_cb_conf)
                assert isinstance(new_cb, CustomModelCheckpoint)
                new_cb.set_registry(model_checkpoint_registry)
                callbacks.append(new_cb)
        elif isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))
        else:
            raise ValueError

    return callbacks


def build_loggers(cfg: DictConfig) -> List[LLogger]:
    """Instantiates loggers from config."""
    loggers: List[LLogger] = []

    if not hasattr(cfg, "loggers"):
        logger.warning("No logger configs found! Skipping...")
        return loggers

    if not isinstance(cfg.loggers, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in cfg.loggers.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers


def build_datamodule(cfg, cache: Dict[str, Any]):
    logger.info("Lightning Data Module %s", cfg.data._target_)
    _cls: BaseDataModule = hydra.utils.get_class(cfg.data._target_)
    data_module = _cls(cfg, cache=cache)
    return data_module


def build_task(cfg: DictConfig) -> BaseTask:
    logger.info("Lightning Module/Task: %s", cfg.task._target_)
    _cls = hydra.utils.get_class(cfg.task._target_)
    task: BaseTask = _cls(cfg=cfg)

    if cfg.debug.hpu.detect_dynamic_ops > 0:
        logger.warning(
            "ENABLED HPU DYNAMIC OPS DEBUGGING, WHICH COULD HEAVILY SLOW DOWN TRAINING PROGRESS!!!"
        )
        from habana_frameworks.torch.utils.experimental import (
            detect_recompilation_auto_model,
        )

        task = detect_recompilation_auto_model(task)
    return task


@rank_zero_only
def get_rich_cfg_tree(
    cfg: DictConfig,
    ordering: Sequence[str] = (
        "cv",
        "task",
        "model",
        "data",
        "loss",
        "trainer",
        "optim",
        "scheduler",
        "loader",
        "env",
        "callbacks",
        "loggers",
    ),
    resolve: bool = True,
    save_path: str | None = None,
    style: str = "dim",
) -> rich.tree.Tree:
    """Tree-based string representation of a DictConfig using the Rich library."""
    cfg = OmegaConf.to_container(cfg, resolve=resolve)

    keys = []
    # add fields from `print_order` to queue
    for k in ordering:
        if k in cfg:
            keys.append(k)
    # add all the other fields to queue (not specified in `print_order`)
    for k in cfg:
        if k not in keys:
            keys.append(k)

    def _add_tree_nodes(tree: rich.tree.Tree, data):
        """Recursively adds nodes to the tree based on the YAML data.

        Args:
            tree: The root tree object.
            data: The YAML data.
        """
        if isinstance(data, dict):
            for k, v in data.items():
                node = tree.add(k)
                _add_tree_nodes(node, v)
        # elif isinstance(data, list):
        # for i, e in enumerate(data):
        #     node = tree.add(str(i))
        #     _add_tree_nodes(node, e)
        else:
            tree.label = f"{tree.label}: {data}"

    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    for k in keys:
        node = tree.add(k)
        _add_tree_nodes(node, cfg[k])

    # save config tree to file
    if save_path:
        with open(save_path, "w") as f:
            rich.print(tree, file=f)
    return tree


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def _wrap_func(
        cfg: DictConfig, *args, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        metric_dict = {}
        # execute the task
        try:
            metric_dict = task_func(cfg, *args, **kwargs)
        # things to do if exception occurs
        except Exception as e:
            # save exception to `.log` file
            logger.exception(
                "Exception occur while running task: %s\n%s", e, traceback.format_exc()
            )
            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise e

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            logger.info("Output dir: %s", cfg.env.output_dir)

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    logger.info("Closing wandb!")
                    wandb.finish()
        return metric_dict

    return _wrap_func


@rank_zero_only
def log_hyperparameters(cfg, module, datamodule, trainer) -> None:
    """Log some information such as config, number of params,.. to Lightning Loggers."""
    del datamodule

    if not trainer.logger:
        logger.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams = {}
    hparams["config/raw"] = OmegaConf.to_container(cfg, resolve=False)
    hparams["config/resolve"] = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in module.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in module.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in module.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def _resolve_ckpt_path(
    ckpt_path_or_dir: str, name: str = "best.ckpt", fold_idx: int | None = None
):
    if os.path.isdir(ckpt_path_or_dir):
        _names = os.listdir(ckpt_path_or_dir)
        if name in _names:
            ckpt_path = os.path.join(ckpt_path_or_dir, name)
        elif fold_idx is not None:
            ckpt_path = os.path.join(
                ckpt_path_or_dir, f"fold_{fold_idx}", "ckpts", name
            )
        else:
            ckpt_path = None
    else:
        ckpt_path = ckpt_path_or_dir
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Could not resolve checkpoint path `{ckpt_path}`")
    return ckpt_path


def load_lightning_state_dict(
    model: L.LightningModule,
    ckpt_path: str | None = None,
    cfg: DictConfig | None = None,
):
    """Loading Lightning checkpoint given the Hydra config.
    Load state dict specified by `ckpt_path` if valid/exist,
    otherwise try to load from `cfg.ckpt` as a fallback option.
    Other states such as callbacks, optimizer,.. will not be loaded.
    Note that this function will modify model inplace.
    No other states like optimizer, scheduler, datamodule are ignored.
    If you want to load these states, consider using other
    Pytorch Lightning builtin methods.
    """
    _is_invalid_ckpt_path = ckpt_path is None or ckpt_path == ""
    if _is_invalid_ckpt_path and cfg.ckpt.path is not None:
        ckpt_path = _resolve_ckpt_path(
            cfg.ckpt.path, name=cfg.ckpt.name, fold_idx=cfg.cv.fold_idx
        )
    elif _is_invalid_ckpt_path and cfg.ckpt.path is None:
        logger.warning("No pretrained checkpoint is found, skip loading..")
        return model

    logger.info("Loading state dict from %s", ckpt_path)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # this is critical to properly load inner EMA state dict
    # this will overwrite `state_dict` to the expected EMA weights
    model.on_load_checkpoint(state_dict)
    state_dict = state_dict["state_dict"]
    try:
        # L.LightningModule.load_from_checkpoint() create new LightningModule instance, which is unexpected
        # task.strict_loading = cfg.ckpt.strict
        # model = model.__class__.load_from_checkpoint(ckpt_path, strict=cfg.ckpt.strict)

        # use this instead
        model.load_state_dict(state_dict=state_dict, strict=cfg.ckpt.strict)
    except Exception as e:
        logger.warning(
            "Exception Occur while loading from checkpoint: %s\n%s",
            e,
            traceback.format_exc(),
        )
        logger.warning("Attempt loading state dict in compatible (unsafe) mode..")
        model = misc_utils.load_state_dict(model, state_dict, strict=cfg.ckpt.strict)
    return model
