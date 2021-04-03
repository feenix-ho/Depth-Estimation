import torch
from torch import nn
from torch.nn import functional as F
from performer_pytorch import Performer

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def patching(locations, patch_size=16):
    locs = locations
    locs[:, :2] -= locs[:, :2] % patch_size
    locs[:, 2:] += (patch_size - locs[:, 2:] % patch_size)
    return locs


class KnowledgeFusion(nn.Module):
    '''
    parameters:
        image: N * H * W * C
        patch_size: P
        object_locations: 4 numbers - x, y; top_left, bottom_right of the image
        transformer related parameters:
            dims
            depth = dims.shape[0]
            heads
        3D tensor: object embeddings from NVTransformer
    returns:
        3D tensor: N * ((H/P) * (W/P)) * dim
    '''

    def __init__(self, *, image_size, emb_size, dims, transformer, channels=3, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = dims[0] + emb_size
        self.output_dim = dims[-1]

        num_patches = (image_size[0] / patch_size, image_size[1] / patch_size)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches[0], num_patches[1], self.input_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.input_dim))
        self.transformer = transformer

        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=num_patches[0], p2=num_patches[1]),
            nn.Linear(patch_dim, self.input_dim),
        )

        prev_dim = self.input_dim
        layers = []
        for dim in dims:
            layers.append(self.transformer(
                dim=prev_dim, depth=1, heads=heads, dim_head=dim_head))
            layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.layers = nn.Sequential(*layers)

    def forward_img(self, img, locations, embs):
        locs = normalize(locations, self.patch_size)
        out_shape = img.shape
        out_shape[-1] = self.output_dim
        cnt = torch.zeros(img.shape[:-1])
        result = torch.zeros(out_shape)

        for loc in locs:
            box = img[:, location[0]:location[2],
                      location[1]:location[3], :]
            x = torch.cat((self.to_patch_embedding(box), embs), dim=-1)
            b, n, _ = x.shape

            pos_embs = self.pos_embedding[:, location[0]:location[2], location[1]:location[3]]
            x += rearrange(pos_embs, 'b h w c -> b (h w) c')
            
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x = self.layers(x)

            result[]

    def forward(self, imgs, locations, embs):
        results = []
        locations = torch.cat((locations, torch.Tensor(
            [0, 0, imgs.shape(1), imgs.shape(2)], dtype=torch.int)), dim=1)
        embs = torch.cat((embs[idx], torch.mean(embs[idx], dim=1)), dim=1)

        for idx, img in enumerate(imgs):
            results.append(self.forward_img(img, locations[idx], embs[idx]))

        return torch.cat(results)
