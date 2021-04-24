import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, repeat
from blocks import FusionBlock, KnowledgeFusion
from kornia import filters

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

def compute_loss(self, trimmed=1., num_scale=4, alpha=.5, **kwagrs):
    def align(imgs, masks):
        imgs = rearrange(imgs, 'b h w -> b (h w)')

        t = imgs.median(dim=1)
        s = masks * torch.abs(imgs - t)
        s = s.sum(dim=1) / masks.sum(dim=1)

        return (imgs - t) / s

    aligned_preds = align(preds, masks)
    aligned_targets = align(targets, masks)

    loss = compute_ssi(trimmed=trimmed, **kwargs)
    if alpha > 0.:
        loss += alpha * compute_reg(num_scale=num_scale, **kwargs)
    return loss.mean(dim=0)


class RDNet(nn.Module):
    '''
    Description:
    Params:
    '''

    def __init__(self, image_size, patch_size, knowledge_dims, dense_dims, latent_dim, **kwargs):
        self.patch_size = patch_size
        num_patches = (image_size[0] / patch_size, image_size[1] / patch_size)
        max_patches = num_patches[0] * num_patches[1]

        self.to_patch = Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)',
                                  p1=num_patches[0], p2=num_patches[1])
        self.knowledge = KnowledgeFusion(
            dims=knowledge_dims,
            num_patches=num_patches,
            max_patches=max_patches,
            **kwargs
        )
        self.dense = DensePrediction(
            inp_dim=knowledge_dims[-1],
            hidden_dims=dense_dims,
            out_dim=latent_dim,
            max_patches=max_patches,
            **kwargs
        )
        self.head = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(latent_dim // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, images, embs, locations):
        patches = self.to_patch(images)
        patches = self.knowledge(patches, embs, locations)
        results = self.dense(patches)
        results = self.head(results)

        return results
