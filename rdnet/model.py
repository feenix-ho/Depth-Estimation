import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, repeat
from blocks import FusionBlock, KnowledgeFusion
from kornia import filters


class RDNet(nn.Module):
    '''
    Description:
    Params:
    '''

    def __init__(self, emb_size, ):
        self.emb_size = emb_size

    def forward():

    def compute_ssi(self, preds, targets, masks, trimmed=1.):
        masks = rearrange(masks, 'b h w -> b (h w)')

        errors = torch.abs(aligned_preds - aligned_targets)[masks]
        cutoff = trimmed * masks.sum(dim=1)
        trimmed_errors = torch.sort(errors, dim=1)[:cutoff]
        ssi_trim = trimmed_erorrs.sum

        return ssi_trim / ()

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

    def compute_loss(self, preds, targets, masks, trimmed=1., num_scale=4, alpha=.5):
        def align(imgs, masks):
            imgs = rearrange(imgs, 'b h w -> b (h w)')

            t = imgs.median(dim=1)
            s = masks * torch.abs(imgs - t)
            s = s.sum(dim=1) / masks.sum(dim=1)

            return (imgs - t) / s

        aligned_preds = align(preds, masks)
        aligned_targets = align(targets, masks)

        loss = compute_ssi(preds, targets, masks, trimmed)
        if alpha > 0.:
            loss += alpha * compute_reg(num_scale=num_scale, kwargs)
        return loss.mean(dim=0)
