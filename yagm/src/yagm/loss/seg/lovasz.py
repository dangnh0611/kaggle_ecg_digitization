"""
Ref: https://github.com/DrHB/2nd-place-contrails/blob/master/src_inference1/lovasz.py
"""

from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch import nn

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse


# def lovasz_grad_jarcard(gt_sorted):
#     """
#     Computes gradient of the Lovasz extension w.r.t sorted errors
#     See Alg. 1 in paper
#     """
#     p = len(gt_sorted)
#     gts = gt_sorted.sum()
#     intersection = gts - gt_sorted.float().cumsum(0)
#     union = gts + (1 - gt_sorted).float().cumsum(0)
#     jaccard = 1.0 - intersection / union
#     if p > 1:  # cover 1-pixel case
#         jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
#     return jaccard


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted, beta=1):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    # gts = gt_sorted.sum()
    tp = gt_sorted.sum() - gt_sorted.float().cumsum(0)
    fp = (1 - gt_sorted).float().cumsum(0)
    fn = gt_sorted.float().cumsum(0)

    Fscore = 1 - tp * (1 + beta**2) / (tp * (1 + beta**2) + fn * beta**2 + fp)
    if p > 1:  # cover 1-pixel case
        Fscore[1:p] = Fscore[1:p] - Fscore[0:-1]
    return Fscore


def lovasz_hinge_flat(logits, labels, beta=1):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted, beta=beta)
    # loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
    return loss


def lovasz_hinge(logits, labels, per_sample=False, beta=1, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_sample:
        loss = mean(
            lovasz_hinge_flat(
                *flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore),
                beta=beta
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = lovasz_hinge_flat(
            *flatten_binary_scores(logits, labels, ignore), beta=beta
        )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()  # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(
            lovasz_softmax_flat(
                *flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                only_present=only_present
            )
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = lovasz_softmax_flat(
            *flatten_probas(probas, labels, ignore), only_present=only_present
        )
    return loss


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def lovasz_sym(logits, labels, per_sample=False, beta=1.0):
    return 0.5 * (
        lovasz_hinge(logits, labels, per_sample=per_sample, beta=beta)
        + lovasz_hinge(-logits, 1 - labels, per_sample=per_sample, beta=1.0 / beta)
    )


class LovaszTverskyLoss(nn.Module):
    def __init__(self, beta, per_sample=False, symetric=False):
        super().__init__()
        self.beta = beta
        self.per_sample = per_sample
        self.symetric = symetric

    def forward(self, pred, target):
        if self.symetric:
            return lovasz_sym(pred, target, per_sample=self.per_sample, beta=self.beta)
        else:
            return lovasz_hinge(
                pred, target, per_sample=self.per_sample, beta=self.beta
            )
