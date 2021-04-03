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

import torch
from torch import nn
from torch.nn import functional as F
from performer_pytorch import Performer


def normalize(object_locations, patch_size=16):
    normalized_locations = object_locations
    for location in normalized_locations:
        location[0] -= location[0] % patch_size
        location[1] -= location[1] % patch_size
        location[2] += (patch_size - location[2] % patch_size)
        location[3] += (patch_size - location[3] % patch_size)
    return normalized_locations


class VTransformer(nn.Module):
    def __init__(self, *, image_size, emb_size, num_classes, dim, transformer, channels=3, patch_size=16):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size)
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.inner_dim = dim + emb_size
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, num_patches, self.inner_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.inner_dim))
        self.transformer = transformer

    def forward(self, img, locations, embs, patch_size=16):
        new_locations = normalize(locations, patch_size)
        # results = torch.zeros(img.shape)

        for location in new_locations:
            obj_img = img[:, location[0]:location[2], location[1]:location[3], :]
            x = self.to_patch_embedding(obj_img)
            b, n, _ = x.shape
            
            x += torch.nn.Unfold(self.pos_embedding[:, location[0]:location[2], location[1]:location[3]], 
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
                        
            x = self.transformer(x)



    def forward(self, imgs, locations, embs, patch_size=16):
