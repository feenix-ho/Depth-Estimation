import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

from einops.layers.torch import Rearrange
from performer_pytorch import Performer
from blocks import FusionBlock


def patching(locations, patch_size=16):
    locs = locations
    locs[:, :, :2] -= locs[:, :, :2] % patch_size
    locs[:, :, 2:] += (patch_size - locs[:, :, 2:] % patch_size)
    return locs


class KnowledgeFusion(nn.Module):
    def __init__(self, *, image_size, emb_size, dims, channels=3, patch_size=16, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size[0] / patch_size, image_size[1] / patch_size)
        max_patches = num_patches[0] * num_patches[1]
        patch_dim = channels * patch_size ** 2

        self.to_patch = Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)',
                                  p1=num_patches[0], p2=num_patches[1])
        self.layers = [FusionBlock(
            emb_size, patch_dim, dims[0], max_patches, **kwargs)]

        for idx in range(len(dims) - 1):
            self.layers.append(FusionBlock(
                dims[idx], dims[idx], dims[idx + 1], max_patches, **kwargs))

    def forward(self, imgs, locations, embs):
        b, n, _ = embs.shape
        locs = patching(locations, patch_size=self.patch_size)
        patches = self.to_patch(imgs)
        masks = torch.zeros(patches.shape[:3], dtype=torch.bool)

        for loc in locs:
            for obj_loc in loc:
                masks[:, obj_loc[0]:obj_loc[2], obj_loc[1]:obj_loc[3]] = True

        masks = rearrange(masks, 'b n h w -> (b n) (h w)')
        patches = repeat(patches, 'b h w d -> b n (h w) d', n=n)

        for layer in self.layers:
            patches, embs = layer(patches, embs, masks)

        masks = rearrange(masks, '(b n) p -> b n p', n=n)
        result = (patches * masks).sum(dim=1) / masks.sum(dim=1)
