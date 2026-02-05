import torch
from torch import nn
import pickle
import os


def _check_tensor_nan_or_inf(v):
    if isinstance(v, torch.Tensor):
        return torch.isnan(v).any() or torch.isinf(v).any()
    elif isinstance(v, dict):
        return any(
            [
                torch.isnan(e).any() or torch.isinf(e).any()
                for e in v.values()
                if isinstance(e, torch.Tensor)
            ]
        )
    elif isinstance(v, (list, tuple)):
        return any(
            [
                torch.isnan(e).any() or torch.isinf(e).any()
                for e in v
                if isinstance(e, torch.Tensor)
            ]
        )
    else:
        raise ValueError("Unsupported type:", type(v))


def check_nan_or_inf_hook(module: nn.Module, input, output, save_path=None):
    """
    Useful to debug a network which produce Nan/+-Inf.
    Example usage:
        `torch.nn.modules.module.register_module_forward_hook(check_nan_or_inf_hook)`
    """
    if _check_tensor_nan_or_inf(output) and not _check_tensor_nan_or_inf(input):
        if save_path is not None:
            data = {
                "input": input.cpu(),
                "module": module.cpu(),
                "output": output.cpu(),
            }
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(data, f)
        raise Exception(
            f"NaN detected in output of {module}, class={module.__class__}, is_training={module.training}, input_type={type(input)}, output_type={type(output)}!"
        )
