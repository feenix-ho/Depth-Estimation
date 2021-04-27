import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from blocks import InjectionBlock, ScratchBlock, ReassembleBlock, RefineBlock
from kornia import filters


class KnowledgeFusion(nn.Module):
    '''
    Description:
    Params:
    '''

    def __init__(self, emb_size, dims, max_patches, patch_dim, **kwargs):
        super().__init__()

        self.layers = [InjectionBlock(
            emb_size, patch_dim, dims[0], max_patches, **kwargs)]

        for idx in range(len(dims) - 1):
            self.layers.append(InjectionBlock(
                dims[idx], dims[idx], dims[idx + 1], max_patches, **kwargs))

    def patching(self, locations, patch_size=16):
        locs = locations
        locs[:, :, :2] -= locs[:, :, :2] % patch_size
        locs[:, :, 2:] += (patch_size - locs[:, :, 2:] % patch_size)
        return locs

    def forward(self, patches, embs, locations):
        '''
        Params:
        Return:
        '''
        b, n, _ = embs.shape
        locs = self.patching(locations, patch_size=self.patch_size)
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
        return result


class DensePrediction(nn.Module):
    def __init__(
        self,
        inp_dim,
        hidden_dims,
        out_dim,
        **kwargs
    ):
        super().__init__()

        self.scratch = ScratchBlock(
            hidden_dim=inp_dim,
            **kwargs
        )

        self.reassemble = ReassembleBlock(
            inp_dim=inp_dim,
            out_dim=hidden_dims,
            **kwargs
        )

        self.refine = RefineBlock(
            in_shape=hidden_dims,
            out_shape=out_dim,
            **kwargs
        )

    def forward(self, embs):
        results = self.scratch(embs)
        results = self.reassemble(results)
        results = self.refine(results)

        return results


class RDNet(nn.Module):
    '''
    Description:
    Params:
    '''

    def __init__(self, image_size, patch_size, knowledge_dims, dense_dims, latent_dim, channels=3, **kwargs):
        super().__init__()
        patch_dim = channels * patch_size ** 2
        num_patches = (image_size[0] / patch_size, image_size[1] / patch_size)
        max_patches = num_patches[0] * num_patches[1]

        self.to_patch = Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)',
                                  p1=num_patches[0], p2=num_patches[1])
        self.knowledge = KnowledgeFusion(
            dims=knowledge_dims,
            max_patches=max_patches,
            patch_dim=patch_dim,
            **kwargs
        )
        self.dense = DensePrediction(
            inp_dim=knowledge_dims[-1],
            hidden_dims=dense_dims,
            out_dim=latent_dim,
            max_patches=max_patches,
            num_patches=num_patches,
            **kwargs
        )
        self.head = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2,
                      kernel_size=3, stride=1, padding=1),
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
