import torch


def group_sum(arr, indices):
    max_index = indices.max().item()
    out = torch.zeros(
        (max_index + 1, *arr.shape[1:]), dtype=arr.dtype, device=arr.device
    )
    out.index_add_(dim=0, index=indices, source=arr, alpha=1)
    return out


def group_max(arr, indices):
    max_index = indices.max().item()
    out = torch.zeros(
        (max_index + 1, *arr.shape[1:]), dtype=arr.dtype, device=arr.device
    )
    out.index_reduce_(
        dim=0, index=indices, source=arr, reduce="amax", include_self=False
    )
    return out


def group_min(arr, indices):
    max_index = indices.max().item()
    out = torch.zeros(
        (max_index + 1, *arr.shape[1:]), dtype=arr.dtype, device=arr.device
    )
    out.index_reduce_(
        dim=0, index=indices, source=arr, reduce="amin", include_self=False
    )
    return out


def group_mean(arr, indices):
    max_index = indices.max().item()
    out = torch.zeros(
        (max_index + 1, *arr.shape[1:]), dtype=arr.dtype, device=arr.device
    )
    out.index_reduce_(
        dim=0, index=indices, source=arr, reduce="mean", include_self=False
    )
    return out
