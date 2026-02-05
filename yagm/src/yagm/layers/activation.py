from functools import partial

from torch import nn

__all__ = ["get_act"]


def get_act(name, inplace=False, **kwargs):
    if name == "identity":
        return nn.Identity
    if name == "gelu":
        return partial(nn.GELU, **kwargs)
    elif name == "relu":
        return partial(nn.ReLU, inplace=inplace, **kwargs)
    elif name == "relu6":
        return partial(nn.ReLU6, inplace=inplace, **kwargs)
    elif name == "silu" or name == "swish":
        return partial(nn.SiLU, inplace=inplace, **kwargs)
    elif name == "selu":
        return partial(nn.SELU, inplace=inplace, **kwargs)
    elif name == "hs":
        return partial(nn.Hardswish, inplace=inplace, **kwargs)
    elif name == "sigmoid":
        return partial(nn.Sigmoid, **kwargs)
    elif name == "tanh":
        return partial(nn.Tanh, **kwargs)
    elif name == "mish":
        return partial(nn.Mish, inplace=inplace, **kwargs)
    elif name == "leakyrelu":
        return partial(nn.LeakyReLU, inplace=inplace, **kwargs)
    elif name == "prelu":
        return partial(nn.PReLU, **kwargs)
    elif name == "elu":
        return partial(nn.ELU, inplace=inplace, **kwargs)
    elif name == "glu":
        return partial(nn.GLU)
    else:
        raise NotImplementedError
