import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

__all__ = ["tf_weight_init", "small_embed_init"]


def tf_weight_init(m):
    """
    Pytorch may have bad default weights initialisation.
    Ref:
        https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization
        https://www.kaggle.com/competitions/liverpool-ion-switching/discussion/145256#863764
        https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html
        https://discuss.pytorch.org/t/suboptimal-convergence-when-compared-with-tensorflow-model/5099/52

    """
    classname = m.__class__.__name__
    if "conv" in classname.lower():
        if hasattr(m, "weight"):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        if hasattr(m, "bias"):
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "bias"):
            nn.init.constant_(m.bias, 0.0)


def small_embed_init(m, a=-1e-4, b=1e-4):
    """Ref: https://github.com/BlinkDL/SmallInitEmb"""
    if isinstance(m, (nn.Embedding)):
        logger.info(
            "Apply SmallInitEmbed (https://github.com/BlinkDL/SmallInitEmb) to %s", m
        )
        nn.init.uniform_(m.weight, a=a, b=b)  # SmallInit(Emb)


def initialize_2d_unet_decoder(module):
    """
    Ref: https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/base/initialization.py
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_2d_unet_head(module, bias=0):
    """
    Ref: https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/base/initialization.py
    """
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, bias)


def initialize_3d_unet_decoder(module):
    """
    Ref: https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/base/initialization.py
    """
    for m in module.modules():
        cls_name = m.__class__.__name__.lower()

        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif "batchnorm" in cls_name:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif "linear" in cls_name:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_3d_unet_head(module, weight_method="xavier", bias=0.0):
    """
    Ref: https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/base/initialization.py
    """
    assert weight_method in ["xavier", "zeros"]
    for m in module.modules():
        cls_name = m.__class__.__name__.lower()
        if "linear" in cls_name or "conv" in cls_name:
            if weight_method == "zeros":
                nn.init.constant_(m.weight, 0.0)
            elif weight_method == "xavier":
                nn.init.xavier_uniform_(m.weight)
            else:
                raise ValueError
            if m.bias is not None:
                if isinstance(bias, float):
                    nn.init.constant_(m.bias, bias)
                elif hasattr(bias, "__len__"):
                    assert len(bias) == len(m.bias)
                    for i in range(len(m.bias)):
                        nn.init.constant_(m.bias[i], bias[i])
                else:
                    raise ValueError


def optimal_bias_init_for_binary_classification(head, pos_rates):
    """
    Ref: Is Heuristic Sampling Necessary in Training Deep  Object Detectors?
    Paper: https://arxiv.org/abs/1909.04868
    Suitable for Sigmoid activation only.
    """
    pos_rates = torch.tensor(pos_rates)
    init_bias = -torch.log(1.0 / pos_rates - 1.0)
    assert len(pos_rates) == len(head.bias)

    for i in range(len(head.bias)):
        nn.init.constant_(head.bias[i], init_bias[i])
    logger.debug("Optimal bias init:\n%s", head.bias)
