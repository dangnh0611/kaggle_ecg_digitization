import logging
from typing import List, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "ScalarWeightedLoss",
    "UncertaintyWeightedLoss",
    "GLSLoss",
    "RLWLoss",
    "DWALoss",
    "IGBV1Loss",
    "AdaptiveInverseWeightedLoss",
    "MTLWeightedLoss",
]

logger = logging.getLogger(__name__)


class ScalarWeightedLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.n = len(weights)

    def get_coefs(self):
        return self.weights

    def forward(self, *losses):
        assert len(losses) == self.n
        total_loss = 0
        for weight, loss in zip(self.weights, losses):
            total_loss += weight * loss
        return total_loss


class UncertaintyWeightedLoss(nn.Module):
    """
    Ref:
        - Method = v1
            + https://arxiv.org/abs/1705.07115
            + https://github.com/oscarkey/multitask-learning/blob/master/reports/aml_report_oscar_key.pdf
            + https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/blob/master/src/models/multi_task_learning_model.py
            + https://github.com/oscarkey/multitask-learning/blob/master/multitask-learning/mnisttask/mnist_loss.py
        - Method = v2
            + https://arxiv.org/pdf/1805.06334
            + https://github.com/Mikoto10032/AutomaticWeightedLoss
        - Related Discussions:
            + https://github.com/ranandalon/mtl/issues/4
            + https://github.com/ranandalon/mtl/issues/2
            + https://piccolboni.info/2018/03/a-simple-loss-function-for-multi-task-learning-with-keras-implementation.html

    Note that we should use loss components, each with `reduction=mean` to correctly balance the loss,
    instead of something like `reduction=sum`. See https://github.com/ranandalon/mtl/issues/4 for more details.

    Args:
        n: number of tasks
        init_log_sigmas: initial log(sigma), default to 0.0 <-> sigma = 1.0
        task_coefs: according to original paper, task_coef = 1.0 for classification and 0.5 for regression task
        method: v1 or v2

    Returns:
        A single combined weighted loss
    """

    def __init__(
        self,
        n: int = 2,
        init_log_sigmas: Union[float, List[float]] = 0.0,
        task_coefs: Union[float, List[float]] = 1.0,
        method: str = "v2",
    ):
        super().__init__()
        assert method in ["v1", "v2"]

        if not isinstance(init_log_sigmas, list):
            init_log_sigmas = [init_log_sigmas] * n
        else:
            assert len(init_log_sigmas) == n

        if not isinstance(task_coefs, list):
            task_coefs = [task_coefs] * n
        else:
            assert len(task_coefs) == n

        self.method = method
        self.n = n
        self.task_coefs = torch.nn.Parameter(
            torch.tensor(task_coefs, dtype=torch.float32), requires_grad=False
        )
        self.log_sigmas = torch.nn.Parameter(
            torch.tensor(init_log_sigmas, dtype=torch.float32), requires_grad=True
        )

    def get_coefs(self):
        vars = torch.exp(2 * self.log_sigmas)
        weights = self.task_coefs / vars
        return weights.cpu()

    def forward(self, *losses):
        assert len(losses) == self.n
        vars = torch.exp(2 * self.log_sigmas)
        coefs = self.task_coefs / vars

        if self.method == "v1":
            total_loss = torch.sum(self.log_sigmas)
        elif self.method == "v2":
            total_loss = torch.sum(torch.log(1 + vars))

        for i, loss in enumerate(losses):
            total_loss += coefs[i] * loss
        return total_loss


class GLSLoss(nn.Module):
    """
    Geometric Loss Strategy (GLS)
    """

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, *losses):
        losses = torch.cat([loss.reshape(1) for loss in losses])
        total_loss = torch.pow(losses.prod(), 1.0 / self.n)
        return total_loss


class RLWLoss(nn.Module):
    """
    Random Loss Weighting (RLW)
    """

    def __init__(self, n):
        super().__init__()
        self.n = n

    def get_coefs(self):
        return self.cur_weights.cpu()

    def forward(self, *losses):
        losses = torch.cat([loss.reshape(1) for loss in losses])
        self.cur_weights = F.softmax(
            torch.randn(self.n, requires_grad=False), dim=-1
        ).to(losses)
        total_loss = torch.mul(losses, self.cur_weights).sum()
        return total_loss


class DWALoss(nn.Module):
    """
    Dynamic Weight Averaging (DWA)
    """

    def __init__(
        self, n, temperature=2.0, init_weights=1.0, start_epoch=2, train_only=True
    ):
        super().__init__()
        self.n = n
        self.temperature = temperature
        assert isinstance(start_epoch, int) and start_epoch >= 2
        self.start_epoch = start_epoch
        if not isinstance(init_weights, list):
            init_weights = [init_weights] * n
        self.cur_weights = torch.tensor(
            init_weights, dtype=torch.float32, requires_grad=False
        )
        self._cur_epoch = 0
        self.all_avg_task_losses = []
        self.cur_epoch_task_losses = [[] for _ in range(self.n)]
        self.train_only = train_only

    def set_epoch(self, epoch):
        """
        Epoch count from 0.
        """
        assert self._cur_epoch + 1 == epoch
        self._cur_epoch = epoch
        cur_epoch_avg_losses = torch.tensor(
            [sum(l) / len(l) for l in self.cur_epoch_task_losses], dtype=torch.float32
        )
        self.all_avg_task_losses.append(cur_epoch_avg_losses)

        # update task weights
        if self._cur_epoch >= self.start_epoch:
            w_i = torch.Tensor(
                self.all_avg_task_losses[-1] / self.all_avg_task_losses[-2]
            )
            # just to ensure no grad
            self.cur_weights = (
                self.n * F.softmax(w_i / self.temperature, dim=-1).detach()
            )
        logger.info(
            "DWA set epoch to %d, num_tasks=%d, num_iters=%s, weights=%s",
            epoch,
            len(self.cur_epoch_task_losses),
            [len(l) for l in self.cur_epoch_task_losses],
            self.cur_weights.cpu().numpy().tolist(),
        )
        # reset
        self.cur_epoch_task_losses = [[] for _ in range(self.n)]

    def get_coefs(self):
        return self.cur_weights.cpu()

    def forward(self, *losses):
        # only cache TRAINING loss to compute loss weights
        if self.training or not self.train_only:
            # cache computed losses
            for i, task_loss in enumerate(losses):
                self.cur_epoch_task_losses[i].append(task_loss.item())

        losses = torch.cat([loss.reshape(1) for loss in losses])
        total_loss = torch.mul(losses, self.cur_weights.to(losses)).sum()
        return total_loss


class IGBV1Loss(nn.Module):
    """
    Improvable Gap Balancing V1 (IGB v1)
    Paper: https://arxiv.org/abs/2307.15429
    Code: https://github.com/YanqiDai/IGB4MTL/blob/main/methods/loss_weight_methods.py
    """

    def __init__(
        self,
        n,
        temperature=1.0,
        init_weights=1.0,
        start_epoch=2,
        ref_epoch_idx=-1,
        train_only=True,
    ):
        super().__init__()
        self.n = n
        self.temperature = temperature
        assert isinstance(start_epoch, int) and start_epoch >= 2
        assert isinstance(ref_epoch_idx, int)
        self.start_epoch = start_epoch
        self.ref_epoch_idx = ref_epoch_idx
        if not isinstance(init_weights, list):
            init_weights = [init_weights] * n
        self.cur_weights = torch.tensor(
            init_weights, dtype=torch.float32, requires_grad=False
        )
        self._cur_epoch = 0
        self.all_avg_task_losses = []
        self.cur_epoch_task_losses = [[] for _ in range(self.n)]
        self.train_only = train_only

    def set_epoch(self, epoch):
        """
        Epoch count from 0.
        """
        assert self._cur_epoch + 1 == epoch
        self._cur_epoch = epoch
        cur_epoch_avg_losses = torch.tensor(
            [sum(l) / len(l) for l in self.cur_epoch_task_losses], dtype=torch.float32
        )
        self.all_avg_task_losses.append(cur_epoch_avg_losses)
        # reset
        self.cur_epoch_task_losses = [[] for _ in range(self.n)]

    def get_coefs(self):
        return self.cur_weights.cpu()

    def forward(self, *losses):
        # cache computed losses
        # only cache TRAINING loss to compute loss weights
        if self.training or not self.train_only:
            for i, task_loss in enumerate(losses):
                self.cur_epoch_task_losses[i].append(task_loss.item())

        losses = torch.cat([loss.reshape(1) for loss in losses])
        if self._cur_epoch >= self.start_epoch:
            gaps = losses.detach() / self.all_avg_task_losses[self.ref_epoch_idx].to(
                losses
            )
            self.cur_weights = (
                self.n * F.softmax(gaps / self.temperature, dim=-1)
            ).detach()
        total_loss = torch.mul(losses, self.cur_weights.to(losses)).sum()
        return total_loss


class AdaptiveInverseWeightedLoss(nn.Module):
    """An Adhoc algorithm.
    Adaptively calculate loss weight depend on each batch's losses.
    The higher loss, the smaller weight
    """

    def __init__(
        self, n, init_weights=1.0, start_epoch=1, mode="per_iter", train_only=True
    ):
        super().__init__()
        assert mode in ["per_epoch", "per_iter"]
        self.mode = mode
        self.n = n
        assert isinstance(start_epoch, int) and start_epoch >= 1
        self.start_epoch = start_epoch

        if not isinstance(init_weights, list):
            init_weights = [init_weights] * n
        self.cur_weights = torch.tensor(
            init_weights, dtype=torch.float32, requires_grad=False
        )
        self._cur_epoch = 0
        self.all_avg_task_losses = []
        self.cur_epoch_task_losses = [[] for _ in range(self.n)]
        self.train_only = train_only

    def _calc_weights_from_losses(self, losses):
        # _sum = losses.sum()
        # weights = 1.0 / losses
        # weights = weights / weights.sum()  # sum up to 1.0
        # weights = weights * _sum / (
        #     weights * losses).sum()  # sum up to original total loss scale

        # the above is equivalent to
        weights = losses.sum() / losses / self.n
        return weights.detach()

    def set_epoch(self, epoch):
        """
        Epoch count from 0.
        """
        assert self._cur_epoch + 1 == epoch
        self._cur_epoch = epoch
        cur_epoch_avg_losses = torch.tensor(
            [sum(l) / len(l) for l in self.cur_epoch_task_losses], dtype=torch.float32
        )
        self.all_avg_task_losses.append(cur_epoch_avg_losses)
        # calculate new weights
        if self.mode == "per_epoch" and self._cur_epoch >= self.start_epoch:
            self.cur_weights = self._calc_weights_from_losses(cur_epoch_avg_losses)

        # reset
        self.cur_epoch_task_losses = [[] for _ in range(self.n)]

    def get_coefs(self):
        return self.cur_weights.cpu()

    def forward(self, *losses):
        # cache computed losses
        # only cache TRAINING loss to compute loss weights
        if self.training or not self.train_only:
            for i, task_loss in enumerate(losses):
                self.cur_epoch_task_losses[i].append(task_loss.item())

        losses = torch.cat([loss.reshape(1) for loss in losses])
        if self.mode == "per_iter":
            self.cur_weights = self._calc_weights_from_losses(losses.detach())
        total_loss = torch.mul(losses, self.cur_weights.to(losses)).sum()
        return total_loss


class MTLWeightedLoss(nn.Module):
    MTL_WEIGHT_METHODS = [
        "scalar",
        "uncertainty_v1",
        "uncertainty_v2",
        "gls",
        "rlw",
        "dwa",
        "inv_per_iter",
        "inv_per_epoch",
        "igb_v1",
    ]

    def __init__(
        self,
        weight_method,
        loss_weights: List[float],
        loss_names: list | None = None,
        task_coefs: list[float] = None,
        scale_invariant=False,
    ):
        """Multi-tasks weighted loss.
        Args:
            weight_method: MTL method to be used
            loss_names: names of lossses, for logging purpose only
            loss_weights: used for scalar weighting method, and only loss with weight > 0
                is considered to compute total loss (weight < 0 is ignored).
            task_coefs: used in uncertainty weighting method,
                usually, task_coef = 1.0 for classification and 0.5 for regression task
            scale_invariant: Scale Invariant (SI) by apply a simple log(),
                ref: https://arxiv.org/abs/2307.15429 (UAI'23)
        """
        super().__init__()
        assert weight_method in self.MTL_WEIGHT_METHODS
        if scale_invariant:
            # Scale Invariant (SI) by apply a simple log()
            # Ref: https://arxiv.org/abs/2307.15429 (UAI'23)
            self.si = lambda losses: [torch.log(loss) for loss in losses]
        else:
            self.si = lambda losses: losses

        if loss_names is None:
            loss_names = [f"loss_{i}" for i in range(len(loss_weights))]
        else:
            assert len(loss_names) == len(loss_weights)
        self.loss_weights = loss_weights
        self.loss_names = loss_names

        self.active_loss_idxs = [i for i, w in enumerate(loss_weights) if w > 0.0]
        self.num_active_losses = len(self.active_loss_idxs)
        logger.info(
            "[MTL] %d/%d active losses (%s) with method=%s weights=%s coefs=%s",
            self.num_active_losses,
            len(loss_weights),
            [self.loss_names[idx] for idx in self.active_loss_idxs],
            weight_method,
            loss_weights,
            task_coefs,
        )

        if weight_method.startswith("uncertainty"):
            _, method = weight_method.split("_")
            assert _ == "uncertainty"
            self.weighting_layer = UncertaintyWeightedLoss(
                n=self.num_active_losses,
                init_log_sigmas=0.0,
                task_coefs=[task_coefs[j] for j in self.active_loss_idxs],
                method=method,
            )
        elif weight_method == "gls":
            self.weighting_layer = GLSLoss(n=self.num_active_losses)
        elif weight_method == "rlw":
            self.weighting_layer = RLWLoss(n=self.num_active_losses)
        elif weight_method == "dwa":
            self.weighting_layer = DWALoss(
                n=self.num_active_losses,
                temperature=2.0,
                init_weights=1.0,
                start_epoch=2,
            )
        elif weight_method == "inv_per_iter":
            self.weighting_layer = AdaptiveInverseWeightedLoss(
                n=self.num_active_losses,
                init_weights=1.0,
                start_epoch=1,
                mode="per_iter",
            )
        elif weight_method == "inv_per_epoch":
            self.weighting_layer = AdaptiveInverseWeightedLoss(
                n=self.num_active_losses,
                init_weights=1.0,
                start_epoch=1,
                mode="per_epoch",
            )
        elif weight_method == "igb_v1":
            self.weighting_layer = IGBV1Loss(
                n=self.num_active_losses,
                temperature=1.0,
                init_weights=1.0,
                start_epoch=2,
                ref_epoch_idx=-1,
            )
        elif weight_method == "igb_v2":
            raise NotImplementedError
        elif weight_method == "scalar":
            self.weighting_layer = ScalarWeightedLoss(
                [loss_weights[j] for j in self.active_loss_idxs]
            )
        else:
            raise ValueError(f"weight_method={weight_method} weights={loss_weights}")

    def get_coefs(self):
        if hasattr(self.weighting_layer, "get_coefs"):
            return self.weighting_layer.get_coefs()
        else:
            return None

    def set_epoch(self, epoch):
        if hasattr(self.weighting_layer, "set_epoch"):
            self.weighting_layer.set_epoch(epoch)

    def forward(self, *losses) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            *losses: all MTL loss terms
        Returns:
            A tuple of (total loss, all losses)
        """
        active_losses = [losses[idx] for idx in self.active_loss_idxs]
        losses = self.si(active_losses)
        total_loss = self.weighting_layer(*losses)
        return total_loss, losses
