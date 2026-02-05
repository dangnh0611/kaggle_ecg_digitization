import logging

from torch import nn

logger = logging.getLogger(__name__)

__all__ = ["BackboneMixin"]


class BackboneMixin(nn.Module):
    """
    Backbone interface for extracting multi-scale feature maps
    For VIT-like variants, it can output same-stride feature maps
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def output_channels(self):
        raise NotImplementedError

    @property
    def output_strides(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError
