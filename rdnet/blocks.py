import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def patching(locations, patch_size=16):
    locs = locations
    locs[:, :, :2] -= locs[:, :, :2] % patch_size
    locs[:, :, 2:] += (patch_size - locs[:, :, 2:] % patch_size)
    return locs


class FusionBlock(nn.Module):
    '''
    Description:
        Perform cross-attention between embeddings before fusing with image patches:
        - Cross-attention module will learn the relational information between objects
        - Visual - relationship fusion will inject knowledge from relation into patches
        through concatenation/sum
    Params:
        - emb_size: dimension of embedding tensor
        - inp_dim: dimension of input patches
        - out_dim: dimension of patches to be output
        - max_patches: maximum number of patches that could be fed
        - readout: type of readout (ignore/add/proj)
        - transformer: the class of transformer to be used
    '''

    def __init__(self, emb_size, inp_dim, out_dim, max_patches, readout, transformer, **kwargs):
        super().__init__()
        self.readout = readout
        self.rel_trans = nn.Sequential([nn.Linear(emb_size, out_dim),
                                        transformer(dim=out_dim, depth=1, **kwargs)])

        self.proj = nn.Linear(inp_dim, out_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_patches + 1, out_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_dim))

        self.transformer = transformer(
            dim=out_dim, depth=1, **kwargs)
        self.project = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim), nn.GELU())

    def get_readout(patches):
        if "ignore" in self.readout:
            return patches[:, 1:]
        elif "add" in self.readout:
            return patches[:, 1:] + patches[:, 0].unsqueeze(1)
        else:
            readout = patches[:, 0].unsqueeze(1).expand_as(patches[:, 1:])
            features = torch.cat((patches[:, 1:], readout), -1)
            return self.project(features)

    def forward(self, imgs, embs, masks):
        '''
        Params:
            - B is number of instances in batch, N is number of objects for each instance,
            P is the number of patches in each instance
            - C is the dimension of each embedding vector and D is the dimension of each patch 
            - imgs: Batch of patches with shape BxNxPxD
            - embs: Batch of embeddings with shape BxNxC
            - masks: Batch of masks to be processed with shape (BxN)xP
        Return:
            - Processed patches
            - Processed embeddings
        '''
        b, n, p, _ = imgs.shape
        x = self.rel_trans(embs)
        x = repeat(x, 'b n () d -> (b n) p d', p=p)

        y = self.proj(imgs)
        y = rearrange(y, 'b n p d -> (b n) p d')
        y += x

        cls_token = repeat(self.cls_token, '() p d -> b p d', b=b*n)
        y = torch.cat([cls_token, y], dim=1)
        y += self.pos_emb[:, :(p + 1)]

        y = self.transformer(y, masks=masks)
        y = self.get_readout(y)
        x = y.mean(dim=1)
        x = rearrange(x, '(b n) d -> b n d', n=n)
        y = rearrange(y, '(b n) p d -> b n p d', n=n)

        return y, x


class KnowledgeFusion(nn.Module):
    '''
    Description:
    Params:

    '''

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
        '''
        Params:kwargs
        Return:
        '''
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
        return result

class ViTBlock(nn.Module):
    '''
        Descriptions:
        Params:
    '''

    def __init__(self, num_patches, dim):
        super().__init__()
        num_p

    def forward(self, patches):

