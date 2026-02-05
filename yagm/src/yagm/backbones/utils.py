import gc
import logging

import torch
from timm.data import resolve_data_config
from torch import nn

logger = logging.getLogger(__name__)


def resolve_hf_preprocess_mean_std(model_name):
    processor = AutoImageProcessor.from_pretrained(model_name)
    mean = torch.FloatTensor(processor.image_mean)
    std = torch.FloatTensor(processor.image_std)
    if processor.do_rescale:
        mean = mean / float(processor.rescale_factor)
        std = std / float(processor.rescale_factor)
    return mean, std


def resolve_timm_preprocess_mean_std(model):
    data_config = resolve_data_config({}, model=model)
    logger.debug("Timm model data config: %s", data_config)
    mean = torch.tensor(data_config["mean"], dtype=torch.float32) * 255
    std = torch.tensor(data_config["std"], dtype=torch.float32) * 255
    return mean, std


def resolve_timm_feature_channels(model: nn.Module):
    """Return a list of feature channels of a timm model, fine to coarse"""
    if hasattr(model, "feature_info"):
        feature_info = model.feature_info
        logger.debug(
            "Feature Info:\n%s",
            (
                feature_info.get_dicts()
                if hasattr(feature_info, "get_dicts")
                else feature_info
            ),
        )
        if isinstance(feature_info, list):
            feature_channels = [stage_info["num_chs"] for stage_info in feature_info]
        else:
            feature_channels = feature_info.channels()

    elif hasattr(model, "embed_dims"):
        feature_channels = model.embed_dims
    else:
        feature_channels = None
    return feature_channels


def resolve_timm_feature_strides(model: nn.Module):
    """Return a list of feature strides of a timm model, fine to coarse"""
    if hasattr(model, "feature_info"):
        feature_info = model.feature_info
        logger.debug(
            "Feature Info:\n%s",
            (
                feature_info.get_dicts()
                if hasattr(feature_info, "get_dicts")
                else feature_info
            ),
        )
        if isinstance(feature_info, list):
            feature_strides = [e["reduction"] for e in model.feature_info]
        else:
            feature_strides = feature_info.reduction()
    else:
        feature_strides = None
    return feature_strides


def prune_timm_backbone(model: nn.Module, prune_norm=False, prune_head=True):
    """Reset classifier attached to a timm model"""
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=0)
    if hasattr(model, "prune_intermediate_layers"):
        try:
            model.prune_intermediate_layers(
                prune_norm=prune_norm,
                prune_head=prune_head,
            )
        except Exception as e:
            logger.warning('Exception occur while prune_intermediate_layers():\n%s', e)
    if hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Module):
        del model.head
        gc.collect()
        torch.cuda.empty_cache()
