'''
    parameters:
        3D tensor: object embeddings from object detection module
        transformer related parameters:
            dim
            depth
            heads

    returns:
        3D tensor: object embeddings
'''

import torch
from torch import nn
from torch.nn import functional as F
from performer_pytorch import Performer

class NVTransformer(nn.Module):
    def __init__(self, emb_size, dims, heads=8, dim_head=64):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = emb_size

        for dim in dims:
            self.layers.append(Performer(dim=prev_dim, depth=1, heads=heads, dim_head=dim_head))
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim

    def forward(self, embs):
        result = embs
        for layer in self.layers:
            result = layer(result)
        return result