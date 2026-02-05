import gc
import logging
from typing import Any, Dict, List, Tuple

import hydra
import lightning.pytorch as pl
import torch
from lightning import Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from timm.scheduler.scheduler import Scheduler as TimmScheduler
from timm.utils.distributed import distribute_bn
from torch import nn
from yagm.metrics import MetricsTracker
from yagm.optim import create_optimizer
from yagm.scheduler import create_scheduler
from yagm.utils import misc as misc_utils
from yagm.utils.model_ema import EMAContainer

import yagm

logger = logging.getLogger(__name__)


class BaseTask(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.steps_per_epoch = cfg.loader.steps_per_epoch

        # build loss function
        self.loss_func = self.build_loss_func()

        # if `configure_model` hook is implemented, override `build_model` to return None
        self.model = self.build_model()

        # NOTE: do not include model ema as a sub module of this
        # unless model_ema will be contained in state_dict()
        self.ema_container: EMAContainer = None

        # torch compiled
        self._is_torch_compiled = False

        # to store cached step results
        # user should reset (clear) this at the end of validation/test epoch
        self.train_step_outputs = {}
        self.eval_step_outputs = {}
        self.predict_step_outputs = []

        self.metrics_tracker = MetricsTracker(
            cfg.task.metrics, keep_top_k=cfg.task.metric_keep_top_k
        )

    def build_loss_func(self):
        loss_func = hydra.utils.instantiate(self.cfg.loss)
        logger.info("--------LOSS FUNCTION--------\n%s", loss_func)
        return loss_func

    def build_model(self):
        model_cls = hydra.utils.get_object(self.cfg.model._target_)
        model = model_cls(self.cfg)

        if getattr(self.cfg.misc, "log_model", True):
            logger.info("--------MODEL--------\n%s", model)
        return model

    # def configure_model(self):
    #     # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.ModelHooks.html#lightning.pytorch.core.hooks.ModelHooks.configure_model
    #     # https://lightning.ai/docs/pytorch/stable/advanced/compile.html#apply-torch-compile-in-configure-model
    #     # https://github.com/Lightning-AI/pytorch-lightning/tree/4e3cf67337cbde346865a3bd7b1322fab710fbf3/examples/fabric/fp8_distributed_transformer#a-note-on-torchcompile
    #     # Hook to create modules in a strategy and precision aware context.
    #     # This is particularly useful for when using sharded strategies (FSDP and DeepSpeed),
    #     # where we’d like to shard the model instantly to save memory and initialization time.
    #     # For non-sharded strategies, you can choose to override this hook or to initialize
    #     # your model under the init_module() context manager.
    #     # This hook is called during each of fit/val/test/predict stages in the same process,
    #     # so ensure that implementation of this hook is idempotent, i.e., after the first time
    #     # the hook is called, subsequent calls to it should be a no-op.
    #     if self.model is not None:
    #         return
    #     logger.info('CONFIGURE MODEL..')
    #     with Trainer.init_module():
    #         self.model = self.build_model()

    @property
    def current_exact_epoch(self):
        return self.global_step / self.steps_per_epoch

    def on_train_start(self) -> None:
        self._configure_ema()
        self._configure_torch_compile()
        self.train_models = [("_self_", self)]
        return super().on_train_start()

    def on_validation_start(self) -> None:
        self._configure_torch_compile()
        self.val_models = self.get_stage_models("val")
        return super().on_validation_start()

    def on_test_start(self) -> None:
        self._configure_torch_compile()
        self.test_models = self.get_stage_models("test")
        return super().on_test_start()

    def on_predict_start(self):
        self._configure_torch_compile()
        return super().on_predict_start()

    def get_batch_size(self, batch):
        return None

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        step_output, step_metrics = self.shared_step(
            self, "train", batch, batch_idx, dataloader_idx=dataloader_idx
        )
        self.log_dict(
            step_metrics,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.get_batch_size(batch),
        )
        # this will significant slow down the training progress, for debug only
        # ref: https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Dynamic_Shapes.html#detecting-dynamic-ops
        if (
            self.cfg.debug.hpu.detect_dynamic_ops > 0
            and batch_idx + 1 >= self.cfg.debug.hpu.detect_dynamic_ops
        ):
            self.analyse_dynamicity()  # Call this after a few steps to generate the dynamicity report
            exit()
        return step_output

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self._shared_eval_step(
            self.val_models, "val", batch, batch_idx, dataloader_idx=dataloader_idx
        )

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self._shared_eval_step(
            self.test_models, "test", batch, batch_idx, dataloader_idx=dataloader_idx
        )

    def state_dict(
        self,
        destination=None,
        prefix="",
        keep_vars=False,
    ):
        full_state_dict = super().state_dict(destination, prefix, keep_vars)
        exclude_prefixes = getattr(self.model, "state_dict_exclude_prefixes", [])
        if exclude_prefixes:
            return {
                k: v
                for k, v in full_state_dict.items()
                if not any(
                    [k.startswith(f"model.{prefix}") for prefix in exclude_prefixes]
                )
            }
        else:
            return full_state_dict

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "ema_state_dict" in checkpoint:
            if not self.cfg.ema.enable:
                logger.warning(
                    "EMA not enabled by config, but found in loaded checkpoint."
                )
            avai_ema_decays = list(checkpoint["ema_state_dict"].keys())
            logger.info(
                "Loading checkpoint: Found %d ema decays in checkpoint: %s",
                len(avai_ema_decays),
                avai_ema_decays,
            )
            # best ckpt is dertermined by best validation result
            # so load the "main" validation model
            # @TODO - load the best model specified in checkpoint itself
            load_ema_decay = self.cfg.ema.val_decays[0]
            if load_ema_decay != 0:
                logger.info(
                    "Attempt load state dict from EMA with decay=%f", load_ema_decay
                )
                checkpoint["state_dict"] = checkpoint["ema_state_dict"][load_ema_decay]
            else:
                logger.info("Loading original state dict without EMA.")
        else:
            assert (
                not self.cfg.ema.enable or len(self.cfg.ema.train_decays) == 0
            ), "EMA was enable by config, but not found in loaded checkpoint."

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks',
        #  'optimizer_states', 'lr_schedulers', 'MixedPrecision']
        if self.ema_container is not None:
            # actual EMA model, excluding _self_
            save_ema_state_dict = {
                save_ema_decay: self.ema_container.ema_models[
                    save_ema_decay
                ].module.state_dict()
                for save_ema_decay in self.cfg.ema.save_decays
                if save_ema_decay != 0
            }
            logger.info(
                "Added %d EMA state dict to checkpoint for saving: %s",
                len(save_ema_state_dict),
                list(save_ema_state_dict.keys()),
            )
            assert "ema_state_dict" not in checkpoint
            checkpoint["ema_state_dict"] = save_ema_state_dict
            # not to save _self_ (main) state_dict
            if all(
                [save_ema_decay != 0 for save_ema_decay in self.cfg.ema.save_decays]
            ):
                logger.info(
                    "Prune main state dict (_self_) from checkpoint for saving.."
                )
                del checkpoint["state_dict"]
            gc.collect()

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        # no special here, add mores in the future
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def optimizer_step(self, *args, **kwargs):
        # previous version do EMA update every on_train_batch_end()
        # bugs when combined with accumulate_grad_batches > 1
        # EMA model is update faster (smaller decay) than expected
        ret = super().optimizer_step(*args, **kwargs)
        if self.ema_container is not None:
            self.ema_container.update(self, self.global_step + 1)
        return ret

    def lr_scheduler_step(self, scheduler, metric=None) -> None:
        # print('LR SCHEDULER:', self.global_step, self.optimizers().param_groups[0]["lr"])
        if self.is_timm_scheduler:
            scheduler.step_update(self.global_step, metric)
        else:
            return super().lr_scheduler_step(scheduler, metric)

    def on_train_epoch_start(self):
        self._ema_reset()
        return super().on_train_epoch_start()

    def on_train_epoch_end(self):
        print("ON TRAIN EPOCH END!")
        self._update_metrics_tracker(None, ["train"])
        # lookahead
        optimizer = self.optimizers()
        assert not isinstance(optimizer, list)
        if hasattr(optimizer, "sync_lookahead"):
            optimizer.sync_lookahead()

        # timm lr scheduler need .step(epoch = epoch) on epoch end
        if self.is_timm_scheduler:
            # for single scheduler only
            lr_scheduler = self.lr_schedulers()
            metric = None
            metric_name = self.cfg.task.metric
            if metric_name is not None:
                metric = self.trainer.callback_metrics.get(metric_name, None)
            lr_scheduler.step(epoch=self.current_epoch + 1, metric=metric)
            logger.info(
                "Timm Scheduler: step() with metric %s=%s on epoch end",
                metric_name,
                metric,
            )

        # this is not neccessary anymore: https://github.com/Lightning-AI/pytorch-lightning/discussions/13060
        # train_loader = self.trainer.train_dataloader
        # val_loader = self.trainer.val_dataloaders
        # train_dataset = train_loader.dataset
        # val_dataset = val_loader.dataset
        # if hasattr(train_dataset, "set_epoch"):
        #     train_dataset.set_epoch(self.current_epoch + 1)
        # if hasattr(val_dataset, "set_epoch"):
        #     val_dataset.set_epoch(self.current_epoch + 1)
        # if hasattr(train_loader.sampler, 'set_epoch'):
        #     train_loader.sampler.set_epoch(self.current_epoch + 1)
        # if hasattr(val_loader.sampler, 'set_epoch'):
        #     val_loader.sampler.set_epoch(self.current_epoch + 1)

        train_loader = self.trainer.train_dataloader
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(self.current_epoch + 1)

        if hasattr(self.loss_func, "set_epoch"):
            self.loss_func.set_epoch(self.current_epoch + 1)

        self._show_current_metrics("train")
        gc.collect()
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        # borrow code from: https://github.com/huggingface/pytorch-image-models/blob/main/train.py
        # @TODO: support pass method from config
        DIST_BN_METHOD = getattr(self.cfg.trainer, "dist_batchnorm", "reduce")
        assert DIST_BN_METHOD in ["reduce", "broadcast", ""]

        # if sync batchnorm is not used, update dist BN if needed
        if (
            self.trainer.world_size > 1
            and not self.cfg.trainer.sync_batchnorm
            and DIST_BN_METHOD in ["reduce", "broadcast"]
        ):
            logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(
                self, self.trainer.world_size, reduce=(DIST_BN_METHOD == "reduce")
            )
            if self.ema_container is not None:
                self.ema_container.update_dist_bn(
                    self.trainer.world_size, dist_bn_method=DIST_BN_METHOD
                )
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        log_metrics, cached_metadatas = self._shared_on_epoch_end(stage="val")
        self._update_metrics_tracker(cached_metadatas, stages=["val"])
        self._show_current_metrics("val")
        gc.collect()
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        log_metrics, cached_metadatas = self._shared_on_epoch_end(stage="test")
        self._update_metrics_tracker(cached_metadatas, stages=["test"])
        self._show_current_metrics("test")
        gc.collect()
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        optimizer = create_optimizer(self.parameters(), self.cfg.optim)
        logger.info("Optimizer: %s\n%s", optimizer.__class__, optimizer)
        lr_scheduler_dict = create_scheduler(optimizer, self.cfg.scheduler)
        lr_scheduler = lr_scheduler_dict["scheduler"]

        self.is_timm_scheduler = False
        if isinstance(lr_scheduler, TimmScheduler) or hasattr(
            lr_scheduler, "step_update"
        ):
            assert (
                hasattr(lr_scheduler, "step")
                and hasattr(lr_scheduler, "step_update")
                and isinstance(lr_scheduler, TimmScheduler)
            )
            self.is_timm_scheduler = True
            logger.info("is_timm_scheduler: %s", self.is_timm_scheduler)

        logger.info("LR Scheduler: %s\n%s", lr_scheduler.__class__, lr_scheduler_dict)
        # if hasattr(lr_scheduler, "state_dict"):
        #     logger.info("LR Scheduler state_dict(): %s", lr_scheduler.state_dict())
        return [optimizer], [lr_scheduler_dict]

    def get_stage_models(self, stage="val"):
        STAGE2DECAYS = {
            "val": self.cfg.ema.val_decays,
            "test": self.cfg.ema.test_decays,
        }
        ret_models = []
        # if in EMA mode, one can still get main (_self_) model
        # by including 0.0 in list of stage decays
        # e.g, val_decays = [0.0, 0.9999, 0.999, 0.99] then
        # perform validation on 4 models: main model and 3 ema models with corresponding decays
        if self.ema_container is not None:
            decays = STAGE2DECAYS[stage]
            for decay in decays:
                # decay=0.0 means using main model (without EMA)
                if decay == 0:
                    ret_models.append(("_self_", self))
                else:
                    ret_models.append(
                        (f"ema_{decay}", self.ema_container.get_model(decay))
                    )
        else:
            # no EMA, so just return the main model
            ret_models.append(("_self_", self))
        return ret_models

    def _configure_ema(self):
        if not self.cfg.train:
            logger.info("Training is disable, skip creating EMA instances")
            return
        if self.ema_container is not None:
            logger.info("EMA Container is created, skip recreating.")
            return
        train_decays = self.cfg.ema.train_decays
        val_decays = [e for e in self.cfg.ema.val_decays if e != 0]
        test_decays = [e for e in self.cfg.ema.test_decays if e != 0]
        save_decays = [e for e in self.cfg.ema.save_decays if e != 0]
        assert set(val_decays).issubset(set(train_decays)) and set(
            test_decays
        ).issubset(set(train_decays))
        self._ema_reset_last_epoch = 0

        # setup exponential moving average of model weights, SWA could be used here too
        if (
            self.cfg.ema.enable
            and self.cfg.ema.train_decays is not None
            and len(self.cfg.ema.train_decays) > 0
        ):
            logger.info(
                "Setting up EMA models with train_decays=%s, val_decays=%s, test_decays=%s, save_decays=%s",
                train_decays,
                val_decays,
                test_decays,
                save_decays,
            )
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
            self.ema_container = EMAContainer(
                self,
                train_decays=train_decays,
                min_decay=self.cfg.ema.min_decay,
                update_after_step=self.cfg.ema.update_after_step,
                use_warmup=self.cfg.ema.use_warmup,
                warmup_gamma=self.cfg.ema.warmup_gamma,
                warmup_power=self.cfg.ema.warmup_power,
                device=(
                    torch.device(self.cfg.ema.device)
                    if self.cfg.ema.device is not None
                    else None
                ),
                foreach=self.cfg.ema.foreach,
                exclude_buffers=self.cfg.ema.exclude_buffers,
            )

    def _configure_torch_compile(self):
        """
        Torch Compile should be done after DDP, so this function should be
        called in on_train_start(), on_validation_start(),..
        """
        if self._is_torch_compiled:
            return

        cfg = self.cfg
        # @TODO - add more logics to validate the success of compilation
        # seem not posible to check here, but still possible somewhere after the graph compilation triggered
        # https://discuss.pytorch.org/t/how-to-check-pytorch2-model-is-compiled/180476
        if cfg.trainer.compile.mode is not None:
            if cfg.trainer.compile.recompile_limit is not None:
                # set the maximum recompile limit: https://docs.pytorch.org/docs/2.7/generated/torch.compile.html
                torch._dynamo.config.recompile_limit = (
                    cfg.trainer.compile.recompile_limit
                )
            compile_modes = cfg.trainer.compile.mode.split("_")
            if len(compile_modes):
                logger.info("Enable Torch Compile with modes=%s", compile_modes)
            else:
                logger.info("Disable Torch Compile feature")

            if "model" in compile_modes:
                # _self_ and ema models share the same code object despite of deepcopy()
                self._configure_ema()

                # compile_ema_models = (
                #     list(self.ema_container.ema_models.values())
                #     if self.ema_container is not None
                #     else []
                # )

                # just compile what we use
                if self.ema_container is not None:
                    compile_ema_decays = [
                        e
                        for e in list(
                            set(self.cfg.ema.val_decays + self.cfg.ema.test_decays)
                        )
                        if e != 0
                    ]
                    compile_ema_models = [
                        self.ema_container.ema_models[ema_decay]
                        for ema_decay in compile_ema_decays
                    ]
                else:
                    compile_ema_models = []

                compile_kwargs = dict(
                    fullgraph=cfg.trainer.compile.model.fullgraph,
                    dynamic=cfg.trainer.compile.model.dynamic,
                    mode=cfg.trainer.compile.model.mode,
                    backend=(
                        "hpu_backend"
                        if (
                            cfg.trainer.accelerator == "hpu"
                            or "hpu" in cfg.trainer.strategy
                        )
                        else "inductor"
                    ),
                )

                logger.info(
                    "Starting Torch Compile on %d models with kwargs=%s",
                    len(compile_ema_models) + 1,
                    compile_kwargs,
                )
                # compile the _self_ task
                self.compile(**compile_kwargs)
                if cfg.trainer.compile.lightning_like:
                    # other methods as well: https://github.com/Lightning-AI/pytorch-lightning/blob/4e3cf67337cbde346865a3bd7b1322fab710fbf3/src/lightning/pytorch/utilities/compile.py#L57
                    # Default/expected Lightning usage/behaviors: When you pass the LightningModule to the Trainer, it will automatically also compile the ``*_step()`` methods.
                    self.training_step = torch.compile(self.training_step, **compile_kwargs)  # type: ignore[method-assign]
                    self.validation_step = torch.compile(self.validation_step, **compile_kwargs)  # type: ignore[method-assign]
                    self.test_step = torch.compile(self.test_step, **compile_kwargs)  # type: ignore[method-assign]
                    self.predict_step = torch.compile(self.predict_step, **compile_kwargs)  # type: ignore[method-assign]

                # compiles all `forward` method of each ema models
                if self.ema_container is not None:
                    for ema_model in compile_ema_models:
                        # this just compile `forward` method
                        ema_model.compile(**compile_kwargs)

            if "optim" in compile_modes:
                self.optimizer_step = torch.compile(
                    self.optimizer_step,
                    fullgraph=cfg.trainer.compile.optim.fullgraph,
                    dynamic=cfg.trainer.compile.optim.dynamic,
                    mode=cfg.trainer.compile.optim.mode,
                )
            self._is_torch_compiled = True

    def shared_step(
        self, model: nn.Module, stage: str, batch, batch_idx, dataloader_idx=None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a shared step for train/val/test

        Returns:
            step_output:
            step_metrics:
        """
        raise NotImplementedError

    def eval_step_single_model(
        self,
        model: nn.Module,
        batch,
        batch_idx,
        dataloader_idx=None,
        stage="val",
        model_name=None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Perform an eval step on a single model.
        Usecase: during training, user may want to validation/test on
        both current model and multiple EMA models with different decays.

        User should implement this method in subclass

        Returns:
            step_output: step output, must include 'loss' key
            step_save_output: step output to cache, used at on_{stage}_epoch_end() to calculate some metrics
            step_metrics: log metric for this step (for this particular model)
        """
        assert stage in ["val", "test"]
        raise NotImplementedError

    def _shared_eval_step(self, models, stage, batch, batch_idx, dataloader_idx=None):
        ret_step_output = None
        log_dict = {"step": self.global_step}  # metrics to log with automatic logging
        for i, (model_name, model) in enumerate(models):
            step_output, step_save_output, step_metrics = self.eval_step_single_model(
                model,
                batch,
                batch_idx,
                dataloader_idx=dataloader_idx,
                stage=stage,
                model_name=model_name,
            )
            self.eval_step_outputs.setdefault(model_name, []).append(step_save_output)
            # add model name as postfix so that metric logged for each model is unique
            # e.g, val/loss/_self_, val/loss/ema_0.999, val/loss/ema_0.9999
            log_dict.update({f"{k}/{model_name}": v for k, v in step_metrics.items()})
            if i == 0:
                # return loss of the first (primary) model
                ret_step_output = step_output
                # # also, metric of primary model is logged with `/_primary_` postfix
                # log_dict.update({f"{k}/_primary_": v for k, v in step_metrics.items()})
        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.get_batch_size(batch),
        )
        return ret_step_output

    def on_epoch_end_save_metadata(
        self, metadata: Dict[str, Any], stage: str, model_name: str
    ) -> Dict[str, Any]:
        # do nothing
        logger.info(
            "%s epoch %.2f ended, saving metadata for model %s..",
            stage,
            self.current_exact_epoch,
            model_name,
        )
        return {}

    def compute_metrics(
        self, step_outputs, dataset, stage="val", model_name=None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute metrics based on cached step outputs.
        Subclass should override this method.

        Returns:
            metrics: dictionary of computed metrics (without any key postfix)
            metadata: dictionary of metadata
        """
        return {}, {}

    def _shared_on_epoch_end(self, stage="val"):
        # post processing
        # calculate & log metric
        if stage == "val":
            dataset = self.trainer.val_dataloaders.dataset
            step_outputs = self.eval_step_outputs
            primary_model_name = self.val_models[0][0]
        elif stage == "test":
            dataset = self.trainer.test_dataloaders.dataset
            step_outputs = self.eval_step_outputs
            primary_model_name = self.test_models[0][0]
        elif stage == "train":
            dataset = self.trainer.train_dataloader.dataset
            step_outputs = self.train_step_outputs
            # _self_
            primary_model_name = self.train_models[0][0]
        else:
            raise ValueError

        log_metrics = {}
        cached_metadatas = {}
        for model_name, model_val_step_outputs in step_outputs.items():
            metrics, metadata = self.compute_metrics(
                model_val_step_outputs, dataset, stage=stage, model_name=model_name
            )
            log_metrics.update({f"{k}/{model_name}": v for k, v in metrics.items()})
            if model_name == primary_model_name:
                # also, metric of primary model is logged with postfix `/_primary_`
                # log_metrics.update({f"{k}/_primary_": v for k, v in metrics.items()})
                pass
            model_cache_metadata = self.on_epoch_end_save_metadata(
                metadata, stage, model_name
            )
            cached_metadatas[model_name] = model_cache_metadata

        # log dict
        self.log_dict(
            log_metrics,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=None,
        )
        # remember to clear cache
        step_outputs.clear()

        return log_metrics, cached_metadatas

    def _update_metrics_tracker(
        self, cached_metadatas, stages: List[str] = ["val"]
    ) -> None:
        # update metrics tracker
        # Current implementation used logged metrics stored in trainer.callback_metrics
        # and called on on_validation_epoch_end() or on_test_epoch_end()
        # but training metrics is aggregated on_training_epoch_end()
        metrics_to_update = [
            k
            for k in self.trainer.callback_metrics.keys()
            if any([stage in k for stage in stages])
        ]
        not_update_stages = set(["train", "val", "test"]) - set(stages)
        assert not any(
            stage in metric_name
            for metric_name in metrics_to_update
            for stage in not_update_stages
        )
        cur_metrics = {
            k: self.trainer.callback_metrics[k].item() for k in metrics_to_update
        }
        self.metrics_tracker.update(
            cur_metrics, cached_metadatas, self.current_exact_epoch, self.global_step
        )
        logger.info(
            "METRICS TRACKER at epoch=%.2f, step=%d :\n%s",
            self.current_exact_epoch,
            self.global_step,
            self.metrics_tracker.repr_table(top_k=3),
        )
        log_dict = {}
        best_metrics = {
            f"_best_@{k}": v["value"]
            for k, v in self.metrics_tracker.best_metrics.items()
            if any([stage in k for stage in stages]) and v is not None
        }
        log_dict.update(best_metrics)
        last_best_instance_metrics = {
            k: v["value"]
            for k, v in self.metrics_tracker.last_best_instance_metrics.items()
            if v is not None
        }
        log_dict.update(last_best_instance_metrics)

        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=None,
        )

    def _ema_reset(self):
        # Implementation of EMA Reset
        # With small amount of data, model is heavily overfiting on later epochs
        # Moving average over bad model weight will also worsen the EMA performance
        # so, after N epoch, reset the _self_ model weight to a better EMA instance's weight
        # this equals to multi-stages training, where in later stage, we load best EMA checkpoint from previous stage
        if self.current_epoch <= 0:
            return
        metric_name = self.cfg.task.metric
        best_metric = self.metrics_tracker.best_metrics[metric_name]
        if best_metric is None:
            return
        # do reset if current epoch is in scheduling
        # or, after K consecutive epochs without improvement
        if self.ema_container is None:
            return

        ema_reset_from = self.cfg.ema.reset_from
        do_ema_reset = False
        if ema_reset_from is None or ema_reset_from == "none":
            do_ema_reset = False
        elif self.current_epoch in self.cfg.ema.reset_sched_epochs:
            logger.info(
                "[EMA RESET] SCHEDULED with ema_reset_from=%s at current epoch %d by %s",
                ema_reset_from,
                self.current_epoch,
                self.cfg.ema.reset_sched_epochs,
            )
            do_ema_reset = True
        elif (
            self.cfg.ema.reset_on_plateau >= 0
            and (
                self.current_epoch - best_metric["epoch"] - 1
                >= self.cfg.ema.reset_on_plateau
            )
            and (
                self.current_epoch - self._ema_reset_last_epoch
                >= self.cfg.ema.min_reset_interval
            )
        ):
            logger.info(
                "[EMA RESET] PLATEAU with ema_reset_from=%s at current epoch %d since no improvement on %d epochs, from best epoch %d",
                ema_reset_from,
                self.current_epoch,
                self.cfg.ema.reset_on_plateau,
                best_metric["epoch"],
            )
            do_ema_reset = True

        if do_ema_reset:
            self._ema_reset_last_epoch = self.current_epoch

        if do_ema_reset:
            # determine the EMA to load from
            if ema_reset_from == "primary":
                # load from main EMA model, specified by ordering of cfg.ema.val_decays
                primary_ema_decays = [e for e in self.cfg.ema.val_decays if e != 0]
                if primary_ema_decays:
                    primary_ema_decay = primary_ema_decays[0]
                    state_dict = self.ema_container.ema_models[
                        primary_ema_decay
                    ].module.state_dict()
                    logger.info(
                        "[EMA RESET] Loading state dict from previous (primary) EMA with decay=%f",
                        primary_ema_decay,
                    )
                    self.load_state_dict(state_dict, strict=True)
                else:
                    logger.info(
                        "No primary EMA found (ema.val_decays=%s), skipping EMA reset..",
                        self.cfg.ema.val_decays,
                    )
            elif ema_reset_from == "last_best":
                # load from current best EMA model (usually of previous epoch)
                last_best_metric = self.metrics_tracker.last_best_instance_metrics[
                    metric_name
                ]
                best_model_name = last_best_metric["model"]
                logger.info(
                    "[EMA RESET] Loadding state dict from model %s with best metric %s=%f at epoch=%d, step=%d",
                    best_model_name,
                    metric_name,
                    last_best_metric["value"],
                    last_best_metric["epoch"],
                    last_best_metric["step"],
                )
                if best_model_name == "_self_":
                    logger.info(
                        "EMA reset: Skipping reset EMA since best model is _self_"
                    )
                elif best_model_name.startswith("ema_"):
                    best_ema_decay = float(best_model_name[4:])
                    state_dict = self.ema_container.ema_models[
                        best_ema_decay
                    ].module.state_dict()
                    self.load_state_dict(state_dict, strict=True)
                else:
                    raise AssertionError
            elif ema_reset_from == "global_best":
                # load from the best ema model (including one saved in checkpoint on disk)
                best_model_name = best_metric["model"]
                logger.info(
                    "[EMA RESET] Best metric `%s`=%f is of model %s at epoch=%d, step=%d",
                    metric_name,
                    best_metric["value"],
                    best_model_name,
                    best_metric["epoch"],
                    best_metric["step"],
                )
                # load saved best ckpt from storage
                assert len(self.trainer.checkpoint_callbacks) == 1
                ckpt_callback = self.trainer.checkpoint_callback
                best_model_path = ckpt_callback.best_model_path
                best_score = ckpt_callback.best_model_score
                checkpoint = torch.load(best_model_path)
                logger.info(
                    "[EMA RESET] Loaded best checkpoint from %s, best_score=%f, epoch=%d, step=%d",
                    best_model_path,
                    best_score,
                    checkpoint["epoch"],
                    checkpoint["global_step"],
                )
                # assert checkpoint["epoch"] + 1 == best_metric["epoch"]
                # assert checkpoint["global_step"] == best_metric["step"]
                assert best_score == best_metric["value"]

                if best_model_name == "_self_":
                    state_dict = checkpoint["state_dict"]
                elif best_model_name.startswith("ema_"):
                    best_ema_decay = float(best_model_name[4:])
                    try:
                        state_dict = checkpoint["ema_state_dict"][best_ema_decay]
                    except:
                        print(best_ema_decay, checkpoint["ema_state_dict"].keys())
                else:
                    raise AssertionError
                self.load_state_dict(state_dict, strict=True)
            else:
                raise ValueError

    def _show_current_metrics(self, stage="val"):
        metrics_table = yagm.utils.misc.dict_as_table(
            {k: v.item() for k, v in self.trainer.logged_metrics.items()},
            sort_by=lambda x: (len(x[0].split("/")), x[0]),
        )
        logger.info(
            "%s epoch=%.2f step=%d ended with metrics:\n%s",
            stage,
            self.current_exact_epoch,
            self.global_step,
            metrics_table,
        )

    def _resolve_saved_checkpoints(self):
        """
        Collect and parse all saved checkpoints from the trainer, returning their
        metadata in a normalized format.

        This method inspects all `CheckpointCallback` instances attached to the
        PyTorch Lightning trainer and aggregates the paths of their tracked
        `best_k_models`. Each checkpoint filename is expected to follow the naming
        convention:

            "ep={epoch}_step={step}_SOMETHINGELSE.ckpt"

        From these filenames, the function extracts the integer `epoch` and `step`
        values and returns a list of metadata dictionaries.

        If no checkpoints are found (e.g., when running in pure validation/test mode
        without training), the function returns a single metadata entry with
        `epoch=0`, `step=0`, and `ckpt_path=None`.

        Returns:
            list[dict]: A list of checkpoint metadata dictionaries. Each dictionary
            contains:
                - "epoch" (int): The epoch number parsed from the checkpoint name.
                - "step" (int): The global step parsed from the checkpoint name.
                - "ckpt_path" (str or None): Absolute path to the checkpoint file,
                  or None if no checkpoint exists (validation/test-only mode).

        Raises:
            AssertionError: If no checkpoints are found but the configuration
                indicates that training mode is active (`self.cfg.train is True`).
        """
        # grab all saved checkpoints
        best_ckpt_paths = []
        for cb in self.trainer.checkpoint_callbacks:
            best_ckpt_paths.extend(list(cb.best_k_models.keys()))
        best_ckpt_paths = list(set(best_ckpt_paths))

        # list of best checkpoints
        # each checkpoint is identified by its metadata dict
        # with keys: epoch, step
        best_ckpt_metas = []
        if best_ckpt_paths:
            for ckpt_path in best_ckpt_paths:
                ckpt_name = os.path.basename(ckpt_path)
                epoch = int(ckpt_name.split("_")[0].replace("ep=", ""))
                step = int(ckpt_name.split("_")[1].replace("step=", ""))
                best_ckpt_metas.append(
                    {"epoch": epoch, "step": step, "ckpt_path": ckpt_path}
                )
        else:
            # in case of validation/test
            assert not self.cfg.train
            best_ckpt_metas.append({"epoch": 0, "step": 0, "ckpt_path": None})
        return best_ckpt_metas

    def get_single_fold_results(self):
        return {}

    def oof_eval(self, all_fold_results, cache=None):
        return {}
