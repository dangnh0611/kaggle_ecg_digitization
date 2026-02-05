"""Exponential Moving Average (EMA) of model updates

Hacked together by / Copyright 2020 Ross Wightman
"""

import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from timm.utils.distributed import distribute_bn

__all__ = ["ModelEmaV3", "EMAContainer"]

logger = logging.getLogger(__name__)


class ModelEma:
    """Model Exponential Moving Average (DEPRECATED)

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This version is deprecated, it does not work with scripted models. Will be removed eventually.

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device="", resume=""):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, "module")
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert isinstance(checkpoint, dict)
        if "state_dict_ema" in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict_ema"].items():
                # ema model may have been wrapped by DataParallel, and need module prefix
                if self.ema_has_module:
                    name = "module." + k if not k.startswith("module") else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            _logger.info("Loaded state_dict_ema")
        else:
            _logger.warning(
                "Failed to find state_dict_ema, starting from loaded model weights"
            )

    def update(self, model):
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, "module") and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = "module." + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1.0 - self.decay) * model_v)


class ModelEmaV2(nn.Module):
    """Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ModelEmaV3(nn.Module):
    """Model Exponential Moving Average V3

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V3 of this module leverages for_each and in-place operations for faster performance.

    Decay warmup based on code by @crowsonkb, her comments:
      If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
      good values for models you plan to train for a million or more steps (reaches decay
      factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
      you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
      215.4k steps).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(
        self,
        model,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_warmup: bool = False,
        warmup_gamma: float = 1.0,
        warmup_power: float = 2 / 3,
        device: Optional[torch.device] = None,
        foreach: bool = True,
        exclude_buffers: bool = False,
    ):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.foreach = foreach
        self.device = device  # perform ema on different device from model if set
        self.exclude_buffers = exclude_buffers
        if self.device is not None and device != next(model.parameters()).device:
            self.foreach = False  # cannot use foreach methods with different devices
            self.module.to(device=device)

    def get_decay(self, step: Optional[int] = None) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        if step is None:
            return self.decay

        # original timm's implementation
        # step = max(0, step - self.update_after_step - 1)

        # just negligible change
        step = max(0, step - self.update_after_step)

        if step <= 0:
            return 0.0

        if self.use_warmup:
            decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
            decay = max(min(decay, self.decay), self.min_decay)
        else:
            decay = self.decay

        return decay

    @torch.no_grad()
    def update(self, model, step: Optional[int] = None):
        decay = self.get_decay(step)
        # print(f"EMA update step={step} decay={decay} device={self.module.device}")
        if self.exclude_buffers:
            self.apply_update_no_buffers_(model, decay)
        else:
            self.apply_update_(model, decay)

    def apply_update_(self, model, decay: float):
        # interpolate parameters and buffers
        if self.foreach:
            ema_lerp_values = []
            model_lerp_values = []
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if ema_v.is_floating_point():
                    ema_lerp_values.append(ema_v)
                    model_lerp_values.append(model_v)
                else:
                    ema_v.copy_(model_v)

            if hasattr(torch, "_foreach_lerp_"):
                torch._foreach_lerp_(
                    ema_lerp_values, model_lerp_values, weight=1.0 - decay
                )
            else:
                torch._foreach_mul_(ema_lerp_values, scalar=decay)
                torch._foreach_add_(
                    ema_lerp_values, model_lerp_values, alpha=1.0 - decay
                )
        else:
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if ema_v.is_floating_point():
                    ema_v.lerp_(model_v.to(device=self.device), weight=1.0 - decay)
                else:
                    ema_v.copy_(model_v.to(device=self.device))

    def apply_update_no_buffers_(self, model, decay: float):
        # interpolate parameters, copy buffers
        ema_params = tuple(self.module.parameters())
        model_params = tuple(model.parameters())
        if self.foreach:
            if hasattr(torch, "_foreach_lerp_"):
                torch._foreach_lerp_(ema_params, model_params, weight=1.0 - decay)
            else:
                torch._foreach_mul_(ema_params, scalar=decay)
                torch._foreach_add_(ema_params, model_params, alpha=1 - decay)
        else:
            for ema_p, model_p in zip(ema_params, model_params):
                ema_p.lerp_(model_p.to(device=self.device), weight=1.0 - decay)

        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(model_b.to(device=self.device))

    @torch.no_grad()
    def set(self, model):
        for ema_v, model_v in zip(
            self.module.state_dict().values(), model.state_dict().values()
        ):
            ema_v.copy_(model_v.to(device=self.device))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


# @TODO - upgrade to ModelEmaV3
class EMAContainer:
    """Container for multiple EMA instances (with different decay) of a model."""

    def __init__(
        self,
        model,
        train_decays=[0.9999],
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_warmup: bool = False,
        warmup_gamma: float = 1.0,
        warmup_power: float = 2 / 3,
        device: Optional[torch.device] = None,
        foreach: bool = True,
        exclude_buffers: bool = False,
    ):
        self.model = model
        self.train_decays = train_decays
        self.ema_models = {
            decay: ModelEmaV3(
                model,
                decay=decay,
                min_decay=min_decay,
                update_after_step=update_after_step,
                use_warmup=use_warmup,
                warmup_gamma=warmup_gamma,
                warmup_power=warmup_power,
                device=device,
                foreach=foreach,
                exclude_buffers=exclude_buffers,
            )
            for decay in self.train_decays
        }
        self.device = (
            None
            if len(self.ema_models) < 1
            else list(self.ema_models.values())[0].module.device
        )
        logger.warning(
            "Created %d ema instances with decays=%s on device %s",
            len(self.train_decays),
            self.train_decays,
            self.device,
        )

    def state_dict(self):
        return {k: v.module.state_dict() for k, v in self.ema_models.items()}

    def to(self, device):
        logger.warning("EMAContainer.to(device) is experimental!")
        for ema_model in self.ema_models.values():
            ema_model.to(device)

    def update(self, model, step: int | None = None):
        for ema_model in self.ema_models.values():
            ema_model.update(model, step)

    def get_models(
        self, decays: Optional[List[float]] = None
    ) -> List[Tuple[str, nn.Module]]:
        decays = decays or self.train_decays
        return {decay: self.ema_models[decay].module for decay in decays}

    def get_model(self, decay):
        return self.ema_models[decay].module

    def update_dist_bn(self, world_size, dist_bn_method="reduce"):
        # update EMA dist BN
        # https://github.com/huggingface/pytorch-image-models/blob/main/train.py#L801
        assert world_size > 1
        for ema_model in self.ema_models.values():
            if "cpu" not in str(self.device):
                if dist_bn_method in ("broadcast", "reduce"):
                    logger.warning(
                        "Distributed BN update with world_size=%d, method=%s",
                        world_size,
                        dist_bn_method,
                    )
                    distribute_bn(
                        ema_model, world_size, reduce=(dist_bn_method == "reduce")
                    )
                else:
                    raise ValueError
