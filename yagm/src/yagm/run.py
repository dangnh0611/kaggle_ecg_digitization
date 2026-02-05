from setproctitle import setproctitle

setproctitle(
    f"/opt/conda/bin/python -m ipykernel_launcher -f /root/.local/share/jupyter/runtime/kernel-hdmt6a3-afe5-44e1-9970-2e87134f1303.json"
)
import numpy as np
import torch

# Setting up custom logging
from yagm.utils.logging import init_logging, setup_logging

init_logging()  # this should be called at the very begining
import logging

logger = logging.getLogger(__name__)

# Register custom resolvers/plugins
from yagm.utils import hydra as hydra_utils

hydra_utils.init_hydra()

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from hydra.types import RunMode
from lightning.pytorch.utilities import disable_possible_user_warnings

torch.set_float32_matmul_precision("medium")  # faster for GPU with bfloat16, e.g A100
logger.warning("WARNING: SET float32_matmul_precision to `medium`")

# ignore all warnings that could be false positives
disable_possible_user_warnings()
import warnings

warnings.filterwarnings("ignore")
import copy
import gc
import math
import os
import pprint

import pandas as pd
from hydra.core.hydra_config import HydraConfig
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from yagm.data.base_data_module import BaseDataModule
from yagm.tasks.base_task import BaseTask
from yagm.utils import lightning as l_utils
from yagm.utils import misc as misc_utils


@l_utils.task_wrapper
def run_one_fold(cfg: DictConfig, CACHE: Dict | None = None) -> Dict[str, Any]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training. This method is wrapped in optional @task_wrapper decorator, that controls the behavior
    during failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: A DictConfig configuration composed by Hydra.

    Returns:
        A dictionary of metrics
    """
    logger.info("Last 5 Git commits:\n%s", "\n".join(cfg.git_commits))
    if cfg.misc.log_raw_cfg:
        logger.info(
            (
                "############## RAW CONFIG ##############\n%s\n"
                "########################################"
            ),
            OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False),
        )

    # https://github.com/pytorch/pytorch/issues/32370
    torch.backends.cudnn.enabled = cfg.trainer.cudnn

    # seed every things, make training deterministic
    if cfg.get("seed") is not None:
        logger.info("Seed everything with seed=%d", cfg.seed)
        # @TODO - MONAI set_determinism() maybe unnecessary
        # try:
        #     from monai.utils import set_determinism

        #     set_determinism(seed=cfg.seed)
        # except:
        #     logger.warning("Unable to set determinism for MONAI!")
        seed_everything(cfg.seed, workers=True)

    ln_callbacks = l_utils.build_callbacks(cfg)
    logger.info(
        "Initialized callbacks:\n%s",
        pprint.pformat([e.__class__ for e in ln_callbacks]),
    )

    if cfg.train:
        ln_loggers = l_utils.build_loggers(cfg)
        logger.info(
            "Initialized loggers:\n%s",
            pprint.pformat([e.__class__ for e in ln_loggers]),
        )
    else:
        ln_loggers = []

    datamodule: BaseDataModule = l_utils.build_datamodule(cfg, cache=CACHE)
    # if cache is empty, build new cache
    if not CACHE:
        logger.info("CACHE is empty! Start caching things..")
        CACHE.update(datamodule.load_cache())

        # YOUR CODE HERE - add more cache if you want
        # ...

        datamodule.set_cache(CACHE)
    logger.info("Using CACHE with keys: %s", list(CACHE.keys()))

    # In most case, steps_per_epoch is -1 and will be resolved from other config
    # Infer steps_per_epoch ahead of time to correctly setup scheduler,..
    # @TODO - https://lightning.ai/docs/pytorch/stable/common/optimization.html#total-stepping-batches
    if cfg.train:
        if cfg.loader.steps_per_epoch <= 0:
            # now we need to access train dataset
            datamodule.prepare_data()
            datamodule.setup(stage="fit")
            actual_steps_per_epoch = misc_utils.calculate_steps_per_epoch(
                len(datamodule.train_dataset),
                cfg.loader.train_batch_size,
                cfg.loader.drop_last,
                cfg.trainer.accumulate_grad_batches,
            )
            logger.warning(
                "Change steps_per_epoch=%d to %d (based on length of train dataloader)",
                cfg.loader.steps_per_epoch,
                actual_steps_per_epoch,
            )
            cfg.loader.steps_per_epoch = actual_steps_per_epoch

        # Resolve epochs and steps related config
        # usually, user want to control training params by epochs, but sometime by steps
        # hence, we need to interpolate/resolve configs related to these 2 terms: epochs and steps
        ##### NOTE 1 #####
        # Lightning's Trainer `max_steps` is not accounted for gradient accumulation
        # i.e, if `max_steps=100` and `accumulate_grad_batches=2`, training is terminated
        # at the 100th batch,but not 200th (100th updates).
        # This make `max_steps` not well aligned with "number of gradient updates" intuitively
        if cfg.trainer.max_epochs is not None and cfg.trainer.max_epochs > 0:
            assert cfg.trainer.max_steps is None or cfg.trainer.max_steps < 0
            # number of gradient updates
            cfg.trainer.max_steps = math.ceil(
                cfg.trainer.max_epochs * cfg.loader.steps_per_epoch
            )
        elif cfg.trainer.max_steps is not None and cfg.trainer.max_steps > 0:
            assert cfg.trainer.max_epochs is None or cfg.trainer.max_epochs < 0
            cfg.trainer.max_epochs = cfg.trainer.max_steps / cfg.loader.steps_per_epoch
        else:
            raise ValueError(
                f"Invalid Trainer config with max_epochs={cfg.trainer.max_epochs} max_steps={cfg.trainer.max_steps}"
            )

    if isinstance(cfg.trainer.val_check_interval, int):
        # user need to specify trainer.val_check_interval in term of number of gradient updates
        # this will interpolate to number of batches to align with default Lightning's behavior
        cfg.trainer.val_check_interval = (
            cfg.trainer.val_check_interval * cfg.trainer.accumulate_grad_batches
        )

    if cfg.optim.base_lr is not None:
        eff_bs = cfg.loader.train_batch_size * cfg.trainer.accumulate_grad_batches
        assert isinstance(cfg.optim.base_lr, float) and cfg.optim.base_lr > 0.0
        if cfg.optim.lr_scale_mode == "sqrt":
            lr = cfg.optim.base_lr * (eff_bs**0.5)
        elif cfg.optim.lr_scale_mode == "linear":
            lr = cfg.optim.base_lr * eff_bs
        else:
            raise ValueError
        logger.info(
            "Adaptive learning rate based on batch size, changed from %f to %f (base_lr=%f, lr_scale_mode=%s, effective_batch_size=%d, train_batch_size=%d, accumulate_grad_batches=%d)",
            cfg.optim.lr,
            lr,
            cfg.optim.base_lr,
            cfg.optim.lr_scale_mode,
            eff_bs,
            cfg.loader.train_batch_size,
            cfg.trainer.accumulate_grad_batches,
        )
        cfg.optim.lr = lr

    if cfg.misc.log_cfg:
        logger.info(
            (
                "############## RESOLVED CONFIG ##############\n%s\n"
                "########################################"
            ),
            OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False),
        )

    if cfg.misc.log_cfg_rich:
        rich_cfg_tree = l_utils.get_rich_cfg_tree(
            cfg,
            resolve=True,
            save_path=os.path.join(cfg.env.fold_output_dir, "cfg.rich"),
        )
        # logger.info(
        #     (
        #         "############## RESOLVED RICH CONFIG ##############\n%s\n"
        #         "########################################"
        #     ),
        #     rich_cfg_tree,
        # )

    # we should build module after data module
    # so that module.configure_optimizers() run with proper finalized config
    task: BaseTask = l_utils.build_task(cfg)

    # setup trainer
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        precision=cfg.trainer.precision,
        logger=ln_loggers,
        callbacks=ln_callbacks,
        fast_dev_run=cfg.trainer.fast_dev_run,
        # max_epochs has higher priority than max_steps
        max_epochs=math.ceil(cfg.trainer.max_epochs) if cfg.train else None,
        min_epochs=cfg.trainer.min_epochs,
        # `max_steps` means number of gradient updates, same as `trainer.global_step`
        # ref: https://github.com/Lightning-AI/pytorch-lightning/discussions/12220
        max_steps=cfg.trainer.max_steps if cfg.train else -1,
        min_steps=cfg.trainer.min_steps,
        max_time=cfg.trainer.max_time,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        limit_predict_batches=cfg.trainer.limit_predict_batches,
        overfit_batches=cfg.trainer.overfit_batches,
        # number of batches instead of number of grad updates :(
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        deterministic=cfg.trainer.deterministic,
        benchmark=cfg.trainer.benchmark,
        inference_mode=cfg.trainer.inference_mode,
        use_distributed_sampler=cfg.trainer.use_distributed_sampler,
        profiler=cfg.trainer.profiler,
        detect_anomaly=cfg.trainer.detect_anomaly,
        barebones=cfg.trainer.barebones,  # all features that may impact raw speed are disabled
        plugins=cfg.trainer.plugins,
        sync_batchnorm=cfg.trainer.sync_batchnorm,
        reload_dataloaders_every_n_epochs=cfg.trainer.reload_dataloaders_every_n_epochs,
        default_root_dir=cfg.trainer.default_root_dir,
    )

    if ln_loggers:
        logger.info("Logging hyperparameters..")
        l_utils.log_hyperparameters(cfg, task, datamodule, trainer)

    logger.info(
        "REPRODUCIBILITY:\ntorch.backends.cudnn.enabled=%s\ntorch.backends.cudnn.deterministic=%s\ntorch.backends.cudnn.benchmark=%s",
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )

    if cfg.train:
        logger.info("Start training..")
        l_utils.load_lightning_state_dict(task, ckpt_path=None, cfg=cfg)
        trainer.fit(
            model=task,
            datamodule=datamodule,
        )

    if cfg.val:
        logger.info("Start validating..")
        ckpt_path = None
        if cfg.train:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path:
                logger.info("Best checkpoint path found in callback: %s", ckpt_path)
            else:
                logger.info("No checkpoint path found in checkpoint callback.")
        l_utils.load_lightning_state_dict(model=task, ckpt_path=ckpt_path, cfg=cfg)
        trainer.validate(model=task, datamodule=datamodule)

    if cfg.test:
        logger.info("Start testing..")
        ckpt_path = None
        if cfg.train:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path:
                logger.info("Best checkpoint path found in callback: %s", ckpt_path)
            else:
                logger.info("No checkpoint path found in checkpoint callback.")
        l_utils.load_lightning_state_dict(model=task, ckpt_path=ckpt_path, cfg=cfg)
        trainer.test(model=task, datamodule=datamodule)

    if cfg.predict:
        logger.info("Start predicting..")
        ckpt_path = None
        if cfg.train:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path:
                logger.info("Best checkpoint path found in callback: %s", ckpt_path)
            else:
                logger.info("No checkpoint path found in checkpoint callback.")
        l_utils.load_lightning_state_dict(model=task, ckpt_path=ckpt_path, cfg=cfg)
        trainer.predict(model=task, datamodule=datamodule)

    # collect results + metrics
    if cfg.train or cfg.val or cfg.test:
        fold_results = task.get_single_fold_results()
        fold_best_metrics = task.metrics_tracker.all_best_metrics
    else:
        fold_results = None
        fold_best_metrics = None

    gc.collect()
    torch.cuda.empty_cache()
    return task, fold_best_metrics, fold_results


@hydra.main(
    version_base="1.3", config_path="../../configs/base/", config_name="run.yaml"
)
def main(cfg: DictConfig) -> Any:
    """Main entry point for training."""
    last_git_commits = misc_utils.get_last_git_logs(5)
    OmegaConf.update(cfg, "git_commits", last_git_commits, force_add=True)
    logger.info("Last 5 Git commits:\n%s", "\n".join(cfg.git_commits))

    # a dict contain some global state which should be cache
    # due to large size, slow loading time (to RAM)
    # typically for loading dataset just once for multiple folds
    CACHE = {}

    # remove lightning logger's default handler
    for name in ["lightning"]:
        target_logger = logging.getLogger(name)
        for handler in target_logger.handlers:
            target_logger.removeHandler(handler)

    # train the model
    fold_idx = cfg.cv.fold_idx
    # train all folds -> eval OOF
    if fold_idx is None:
        all_folds = list(range(cfg.cv.num_folds))
    elif isinstance(fold_idx, int):
        all_folds = [fold_idx]
    else:
        all_folds = fold_idx
        if len(all_folds) < 1:
            raise ValueError

    logger.info(
        "Config cv.fold_idx=%s, training on %d folds: %s",
        fold_idx,
        len(all_folds),
        all_folds,
    )

    all_fold_results = {}
    all_fold_best_metrics = {}

    last_fold_file_handler = None

    # hydra_cfg = HydraConfig.get()
    # logger.info(
    #     "HYDRA CONFIG:\n%s", OmegaConf.to_yaml(hydra_cfg, resolve=True, sort_keys=False)
    # )

    task: BaseTask = None
    for i, fold_idx in enumerate(all_folds):
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg.cv.fold_idx = fold_idx

        # Create new output directory for this specific fold
        fold_output_dir = fold_cfg.env.fold_output_dir
        os.makedirs(fold_output_dir, exist_ok=True)

        # save log for this specific fold in another log file
        fold_log_file_path = os.path.join(
            fold_output_dir,
            f"{HydraConfig.get().job.name}.log",
        )
        if last_fold_file_handler is not None:
            # remove last file handler of previous fold
            assert isinstance(last_fold_file_handler, logging.FileHandler)
            logging.getLogger().removeHandler(last_fold_file_handler)
        last_fold_file_handler = setup_logging(
            HydraConfig.get().job_logging,
            name=None,
            file_path=fold_log_file_path,
            level=logging.getLogger().level,
        )

        logger.info("\n--------START RUN ON FOLD %d--------", fold_idx)
        task, fold_best_metrics, fold_results = run_one_fold(fold_cfg, CACHE)
        if i < len(all_folds) - 1:
            del task
            gc.collect()
            torch.cuda.empty_cache()
        all_fold_results[fold_idx] = fold_results
        all_fold_best_metrics[fold_idx] = fold_best_metrics

        if cfg.train or cfg.val or cfg.test:
            # log the metrics summary of all done folds
            summary_df = misc_utils.create_metrics_summary(
                log_metrics=cfg.task.log_metrics,
                all_fold_best_metrics=all_fold_best_metrics,
                all_oof_best_metrics=None,
                topk=1,
                csv_path=os.path.join(cfg.env.output_dir, "metrics_summary.csv"),
            )

    ########### OOF eval ##########
    if (cfg.train or cfg.val or cfg.test) and cfg.oof_eval and len(all_folds) > 1:
        logger.info("STARTING OOF EVALUATION WITH %d FOLDS", len(all_folds))
        all_oof_best_metrics = task.oof_eval(
            all_fold_results=all_fold_results, cache=CACHE
        )

        # log the metrics summary of all done folds
        summary_df = misc_utils.create_metrics_summary(
            log_metrics=cfg.task.log_metrics,
            all_fold_best_metrics=all_fold_best_metrics,
            all_oof_best_metrics=all_oof_best_metrics,
            topk=1,
            csv_path=os.path.join(cfg.env.output_dir, "metrics_summary.csv"),
        )

    # cleaning
    if last_fold_file_handler is not None:
        assert isinstance(last_fold_file_handler, logging.FileHandler)
        logging.getLogger().removeHandler(last_fold_file_handler)

    ##### LOG SWEEP RESULTS #####
    hydra_cfg = HydraConfig.get()
    # log sweep result to a csv file for easier tracking
    if hydra_cfg.mode == RunMode.RUN:  # RUN
        job_sweep_csv_path = os.path.join(hydra_cfg.run.dir, "job_sweep.csv")
    elif hydra_cfg.mode == RunMode.MULTIRUN:  # MULTIRUN
        job_sweep_csv_path = os.path.join(hydra_cfg.sweep.dir, "job_sweep.csv")
    else:
        raise AssertionError

    if summary_df is not None:
        summary_df = summary_df.sort_values(by="fold", key=lambda x: x != "OOF")
        # Flattening the DataFrame while keeping the column names as concat of row index and original column names
        keep_cols = [col for col in summary_df.columns if col not in ["fold", "rank"]]
        flat_metric_cols = [
            f"{'fold' if fold_idx != 'OOF' else ''}{fold_idx}@{col}"
            for fold_idx in summary_df["fold"]
            for col in keep_cols
        ]
        flat_metric_values = summary_df[keep_cols].values.flatten()
        # Create a DataFrame with the new column names as a single row
        new_flat_df = pd.DataFrame([flat_metric_values], columns=flat_metric_cols)

        # overrided hyperparams
        param_cols = []
        for e in hydra_cfg.overrides.task:
            k = e.split("=")[0]
            v = "=".join(e.split("=")[1:])
            new_flat_df[k] = [v]
            param_cols.append(k)
            assert type(k) == str and type(v) == str
        try:
            sweep_df = pd.read_csv(job_sweep_csv_path)
        except FileNotFoundError:
            sweep_df = pd.DataFrame()

        sweep_df = pd.concat([sweep_df, new_flat_df], ignore_index=True).reset_index(
            drop=True
        )
        # sort param_cols so that sweep params go first
        # prevent mixed types of old and new rows
        sweep_df = sweep_df.astype(str)
        constant_cols = set(sweep_df.columns[sweep_df.nunique() < 2])
        constant_param_cols = sorted(list(set(param_cols).intersection(constant_cols)))
        sweep_param_cols = sorted(list(set(param_cols).difference(constant_cols)))
        metric_cols = list(set(sweep_df.columns).difference(param_cols))
        # sort, keep the original dataframe columns order
        metric_cols = sorted(metric_cols, key=lambda x: list(sweep_df.columns).index(x))
        sweep_df = sweep_df[sweep_param_cols + metric_cols + constant_param_cols]
        # Drop column which contain all nan or "NA" or "nan" values
        sweep_df = sweep_df.loc[
            :,
            (
                ~(
                    sweep_df.isna()
                    | sweep_df.isnull()
                    | sweep_df.eq("NA")
                    | sweep_df.eq("nan")
                    | sweep_df.eq("null")
                )
            ).any(),
        ]
        # race condition could happen, but we simply accept & ignore it
        sweep_df.to_csv(job_sweep_csv_path, index=False)

    del task, CACHE
    gc.collect()
    torch.cuda.empty_cache()

    # sleep for some seconds so that Garbage Collector has time for its job
    # this helps preventing OOM for some case
    import time

    time.sleep(cfg.misc.cooldown_sec)

    return None


if __name__ == "__main__":
    main()
