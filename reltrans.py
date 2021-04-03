import torch
from torch import nn
from torch.nn import functional as F
from performer_pytorch import Performer

class RelationalTransformer(nn.Module):
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
    def __init__(self, emb_size, dims, transformer, heads=8, dim_head=64):
        super().__init__()
        prev_dim = emb_size
        self.transformer = transformer

        layers = []
        for dim in dims:
            layers.append(self.transformer(dim=prev_dim, depth=1, heads=heads, dim_head=dim_head))
            layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.layers = nn.Sequential(*layers)

    def forward(self, embs):
        return self.layers(embs)
