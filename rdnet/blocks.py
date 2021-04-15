import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from performer_pytorch import Performer

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


def patching(locations, patch_size=16):
    locs = locations
    locs[:, :, :2] -= locs[:, :, :2] % patch_size
    locs[:, :, 2:] += (patch_size - locs[:, :, 2:] % patch_size)
    return locs


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


class InjectionBlock(nn.Module):
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
        self.layers = [InjectionBlock(
            emb_size, patch_dim, dims[0], max_patches, **kwargs)]

        for idx in range(len(dims) - 1):
            self.layers.append(InjectionBlock(
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
    def __init__(self, depth, hidden_dim, max_patches, hooks, readout, transformer, **kwargs):

        masks = rearrange(masks, '(b n) p -> b n p', n=n)
        result = (patches * masks).sum(dim=1) / masks.sum(dim=1)
        return result


class ViTBlock(nn.Module):
    '''
        Descriptions:
        Params:
    '''

    def __init__(self, hidden_dim, max_patches, hooks, readout, transformer, **kwargs):
        super().__init__()

        self.pos_emb = nn.Parameter(torch.randn(1, max_patches + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.transformers = nn.ModuleList()

        pre = 0
        for cur in hooks:
            self.transformers.append(transformer(dim = hidden_dim, depth = cur - pre , **kwargs))
            pre = cur

        self.act_postprocesses = nn.ModuleList()

        self.act_postprocesses.append(
            nn.Sequential(
                readout_oper[0],
                Transpose(1, 2),
                nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
                nn.Conv2d(
                    in_channels=vit_features,
                    out_channels=features[0],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=features[0],
                    out_channels=features[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    bias=True,
                    dilation=1,
                    groups=1,
                ),
            )
        )

        self.act_postprocesses.append(
            nn.Sequential(
                readout_oper[1],
                Transpose(1, 2),
                nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
                nn.Conv2d(
                    in_channels=vit_features,
                    out_channels=features[1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=features[1],
                    out_channels=features[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=True,
                    dilation=1,
                    groups=1,
                ),
            )
        )

        self.act_postprocesses.append(
            nn.Sequential(
                readout_oper[2],
                Transpose(1, 2),
                nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
                nn.Conv2d(
                    in_channels=vit_features,
                    out_channels=features[2],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
        )

        self.act_postprocesses.append(
            nn.Sequential(
                readout_oper[3],
                Transpose(1, 2),
                nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
                nn.Conv2d(
                    in_channels=vit_features,
                    out_channels=features[3],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.Conv2d(
                    in_channels=features[3],
                    out_channels=features[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            )
        )

    def forward(self, embs):
        x = embs
        results = []
        for transformer, act_postprocess in zip(self.transformers, self.act_postprocesses):
            x = transformer(x)
            t = act_postprocess[0:2](x)
            results.append(t)

        unflatten = nn.Sequential(
            nn.Unflatten(
                2,
                torch.Size(
                    [
                        h // pretrained.model.patch_size[1],
                        w // pretrained.model.patch_size[0],
                    ]
                ),
            )
        )
        
        for result in results:
            if result.ndim == 3:
                result = unflatten(result)

        for result, act_postprocess in zip(results, self.act_postprocesses):
            result = act_postprocess[3 : len(act_postprocess)](result)

        return result

class ActPostprocessBlock(nn.Module):
    def __init__(self, num_patches, )

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
