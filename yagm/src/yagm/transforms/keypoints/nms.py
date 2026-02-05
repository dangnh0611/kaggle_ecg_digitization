import torch
import time
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)


def _l2_dist_torch(kpts, ref_kpt):
    return torch.sqrt((kpts - ref_kpt[None]).pow(2).sum(dim=-1))


def keypoints_nms(pred, nms_thres, max_dets=None, timeout=None, use_tqdm=False):
    """
    Args:
        pred: coordinate of shape (N, D)
    Returns:
        np.ndarray: indexes to keep.
    """
    if len(pred) == 0:
        return []
    confs = pred[..., -1]
    kpts = pred[..., :-1]
    order = confs.argsort(descending=True)
    keep = []
    # Use tqdm if use_tqdm is True
    pbar = tqdm(desc=f"NMS len={len(kpts)} max_dets={max_dets}") if use_tqdm else None
    start = time.time()
    while len(order) > 0:
        i = order[0].item()
        keep.append(i)
        if pbar:
            pbar.update(1)
        if len(order) == 1:
            break
        if max_dets and len(keep) >= max_dets:
            logger.debug("NMS: max_dets exceeded %d > %d", len(keep), max_dets)
            break
        if timeout:
            time_diff = time.time() - start
            if time_diff > timeout:
                logger.warning(
                    "NMS timeout exceeded %f > %f, num_outputs=%d",
                    time_diff,
                    timeout,
                    len(keep),
                )
                break

        # calculate distances in batch
        dists = _l2_dist_torch(kpts[order[1:]], kpts[i])
        inds = torch.nonzero(dists >= nms_thres)[:, 0]
        order = order[inds + 1]
    if pbar:
        pbar.close()
    return keep
