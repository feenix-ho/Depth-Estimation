import torch
from torch import nn
from torch.nn import functional as F
from performer_pytorch import Performer
from einops import rearrange, repeat


class FusionBlock(nn.Module):
    '''
    Params:
    Return:

    '''

    def __init__(self, emb_size, inp_dim, out_dim, max_patches, transformer, **kwargs):
        super().__init__()
        self.rel_trans = nn.Sequential([nn.Linear(emb_size, out_dim),
                                        transformer(dim=out_dim, depth=1, heads=heads, dim_head=dim_head)])

        self.proj = nn.Linear(inp_dim, out_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_patches, out_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_dim))

        self.transformer = transformer(
            dim=out_dim, depth=1, heads=heads, dim_head=dim_head)

    def forward(self, imgs, embs, masks):
        b, n, p, _ = imgs.shape
        x = self.rel_trans(embs)
        x = repeat(x, 'b n () d -> (b n) p d', p=p)

        y = self.proj(imgs)
        y = rearrange(y, 'b n p d -> (b n) p d')
        y += x

        cls_token = repeat(self.cls_token, '() p d -> b p d', b=b*n)
        y = torch.cat([cls_token, y], dim=1)
        y += self.pos_emb

        y = self.transformer(y, masks=masks)
        y = y[:, :p]
        x = y.mean(dim=1)
        x = rearrange(x, '(b n) p d -> p b n d', n=n)[0]
        y = rearrange(y, '(b n) p d -> b n p d', n=n)

        return y, x

