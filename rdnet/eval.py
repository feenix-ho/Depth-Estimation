import torch
from torch import nn

from einops import rearrange, repeat
from kornia import filters


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def compute_ssi(self, preds, targets, masks, trimmed=1.):
    masks = rearrange(masks, 'b h w -> b (h w)')
    M = masks.sum(dim=1)

    errors = torch.abs(aligned_preds - aligned_targets)[masks]
    trimmed_errors = torch.sort(errors, dim=1)
    ssi_trim = []

    for i in range(M.shape[0]):
        cutoff = trimmed * M[i]
        error = trimmed_errors[i][masks[i]]
        trimmed_error = error[:cutoff].sum()
        ssi_trim.append(trimmed_error / M[i])

    return torch.cat(ssi_trim)


def compute_reg(self, preds, targets, masks, num_scale=4):
    def compute_grad(preds, targets, masks):
        diff = repeat(preds - targets, 'b h w -> b c h w', c=1)
        grads = filters.spatial_gradient(diff)
        abs_grads = torch.abs(grads[:, 0, 0]) + torch.abs(grads[:, 0, 1])
        sum_grads = torch.sum(abs_grads * masks, (1, 2))
        return sum_grads / masks.sum((1, 2))

    total = 0
    step = 1

    for scale in range(num_scale):
        total += compute_grad(preds[:, ::step, ::step],
                              targets[:, ::step, ::step], masks[:, ::step, ::step])
        step *= 2

    return total


def compute_loss(self, preds, targets, masks, trimmed=1., num_scale=4, alpha=.5, **kwagrs):
    def align(imgs, masks):
        masks = rearrange(masks, 'b h w -> b (h w)')
        patches = rearrange(imgs, 'b h w -> b (h w)')

        t = patches.median(dim=1)
        s = masks * torch.abs(patches - t)
        s = s.sum(dim=1) / masks.sum(dim=1)

        return (imgs - t) / s

    aligned_preds = align(preds, masks)
    aligned_targets = align(targets, masks)

    loss = compute_ssi(preds, targets, masks, trimmed)
    if alpha > 0.:
        loss += alpha * compute_reg(preds=preds, targets=targets,
                                    masks=masks, num_scale=num_scale, **kwargs)
    return loss.mean(dim=0)
