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
        3D tensor: N * ((H/P) * (W/P)) * dims[-1]
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
        location[2] += (patch_size - 1 - location[2] % patch_size)
        location[3] += (patch_size - 1 - location[3] % patch_size)
    return normalized_locations

class VTransformer(nn.Module):
    def __init__():

    def forward():

