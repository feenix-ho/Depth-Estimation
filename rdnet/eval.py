import torch
from torch import nn

import numpy as np

from einops import rearrange, repeat
from kornia import filters


def compute_errors(gt, pred):
    dims = (2, 3)
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean(dim)
    d2 = (thresh < 1.25 ** 2).mean(dim)
    d3 = (thresh < 1.25 ** 3).mean(dim)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean(dim))

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean(dim))

    abs_rel = np.mean(np.abs(gt - pred) / gt, dim)
    sq_rel = np.mean(((gt - pred) ** 2) / gt, dim)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2, dim) - np.mean(err, dim) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err, dim)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def compute_ssi(preds, targets, masks, trimmed=1., eps=1e-4):
    masks = rearrange(masks, 'b c h w -> b c (h w)')
    errors = rearrange(torch.abs(preds - targets), 'b c h w -> b c (h w)')
    b, _, n = masks.shape
    valids = masks.sum(2, True)
    invalids = (~masks).sum(2, True)

    errors -= (errors + eps) * (~masks)
    sorted_errors, _ = torch.sort(errors, dim=2)
    assert torch.isnan(sorted_errors).sum() == 0
    idxs = repeat(torch.arange(end=n, device=valids.device),
                  'n -> b c n', b=b, c=1)
    cutoff = (trimmed * valids) + invalids
    trimmed_errors = torch.where((invalids <= idxs) & (
        idxs < cutoff), sorted_errors, sorted_errors - sorted_errors)

    assert torch.isnan(trimmed_errors).sum() == 0
    return (trimmed_errors / valids).sum(dim=2)


def compute_reg(preds, targets, masks, num_scale=4):
    def compute_grad(preds, targets, masks):
        grads = filters.spatial_gradient(preds - targets)
        abs_grads = torch.abs(grads[:, :, 0]) + torch.abs(grads[:, :, 1])
        sum_grads = torch.sum(abs_grads * masks, (2, 3))
        return sum_grads / masks.sum((2, 3))

    total = 0
    step = 1

    for scale in range(num_scale):
        total += compute_grad(preds[:, :, ::step, ::step],
                              targets[:, :, ::step, ::step], masks[:, :, ::step, ::step])
        step *= 2

    return total


def compute_loss(preds, targets, masks, trimmed=1., num_scale=4, alpha=.5, eps=1e-4, **kwargs):
    def align(imgs, masks):
        patches = rearrange(imgs, 'b c h w -> b c (h w)')
        patched_masks = rearrange(masks, 'b c h w -> b c (h w)')
        meds = []

        for img, mask in zip(imgs, masks):
            med = torch.masked_select(img, mask).median(0, True)[0]
            meds.append(med.unsqueeze(1))

        t = repeat(torch.cat(meds), 'b c -> b c d', d=1)
        masked_abs = torch.abs(patches - t) * patched_masks
        assert torch.isnan(masked_abs).sum() == 0

        s = masked_abs.sum(2, True) / patched_masks.sum(2, True) + eps
        try:
            assert 0 not in s
        except:
            print("Masked absolute: ", masked_abs[s[:, :, 0] < eps])
            print("Patches: ", patches[s[:, :, 0] < eps])
            assert False
        assert torch.isnan(s).sum() == 0
        temp = (imgs - t.unsqueeze(3)) / s.unsqueeze(3)
        assert torch.isnan(temp).sum() == 0

        return (imgs - t.unsqueeze(3)) / s.unsqueeze(3)

    assert (preds * preds).sum() > eps
    aligned_preds = align(preds, masks)
    aligned_targets = align(targets, masks)
    assert torch.isnan(aligned_preds).sum() == 0
    assert torch.isnan(aligned_targets).sum() == 0

    loss = compute_ssi(aligned_preds, aligned_targets, masks, trimmed) / 2
    assert torch.isnan(loss).sum() == 0
    if alpha > 0.:
        loss += alpha * compute_reg(aligned_preds, aligned_targets,
                                    masks, num_scale)
    assert torch.isnan(loss).sum() == 0
    return loss.mean(dim=0)
