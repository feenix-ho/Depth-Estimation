import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def _make_fusion_block(features, use_bn, activation):
    return FeatureFusionBlock(
        features,
        activation(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(
            nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)

        return self.project(features)


def get_readout_oper(inp_dim, out_dims, use_readout, start_index=1, **kwargs):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(out_dims)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(out_dims)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(inp_dim, start_index) for dim in out_dims
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return nn.ModuleList(readout_oper)


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
        - use_readout: type of readout (ignore/add/proj)
        - transformer: the class of transformer to be used
    '''

    def __init__(self, emb_size, inp_dim, out_dim, max_patches, use_readout, transformer, landmarks, **kwargs):
        super().__init__()
        self.readout = get_readout_oper(inp_dim=out_dim, out_dims=[
                                        out_dim], use_readout=use_readout, **kwargs)
        self.rel_trans = nn.Sequential(nn.Linear(emb_size, out_dim),
                                       transformer(dim=out_dim, depth=1, num_landmarks=landmarks))
        self.proj = nn.Linear(inp_dim, out_dim)
        self.pos_emb = nn.Parameter(
            torch.randn(1, int(max_patches), out_dim))
        self.cls_token = nn.Parameter(torch.randn(1, out_dim))

        self.transformer = transformer(
            dim=out_dim, depth=1, num_landmarks=landmarks)

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
        x = repeat(x, 'b n d -> (b n) p d', p=p)

        y = self.proj(imgs)
        y = rearrange(y, 'b n p d -> (b n) p d')
        y += x

        cls_token = repeat(self.cls_token, 'p d -> b p d', b=b*n)
        y += self.pos_emb[:, :p]
        y = torch.cat([cls_token, y], dim=1)

        cls_masks = torch.ones((b * n, 1), dtype=torch.bool).to(masks.device)
        masks = torch.cat([cls_masks, mask], dim=1)
        y = self.transformer(y, mask=masks)
        y = self.readout[0](y)

        x = y.mean(dim=1)
        x = rearrange(x, '(b n) d -> b n d', n=n)
        y = rearrange(y, '(b n) p d -> b n p d', n=n)

        return y, x


class ScratchBlock(nn.Module):
    '''
        Descriptions:
        Params:
    '''

    def __init__(self, hidden_dim, max_patches, hooks, use_readout, transformer, landmarks, **kwargs):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(
            1, int(max_patches), hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, hidden_dim))

        self.transformers = nn.ModuleList()

        pre = 0
        for cur in hooks:
            self.transformers.append(transformer(
                dim=hidden_dim, depth=cur - pre, num_landmarks=landmarks))
            pre = cur

    def forward(self, embs):
        b, p, _ = embs.shape
        x = embs + self.pos_emb[:, :p]
        results = []

        cls_token = repeat(self.cls_token, 'p d -> b p d', b=b)
        x = torch.cat([cls_token, x], dim=1)

        for transformer in self.transformers:
            x = transformer(x)
            results.append(x)

        return results


class ReassembleBlock(nn.Module):
    def __init__(self, num_patches, inp_dim, out_dims, start_index=1, **kwargs):
        super().__init__()
        self.reassembles = nn.ModuleList()
        readout_oper = get_readout_oper(
            inp_dim=inp_dim, out_dims=out_dims, **kwargs)

        self.reassembles.append(
            nn.Sequential(
                readout_oper[0],
                Transpose(1, 2),
                nn.Unflatten(2, torch.Size(
                    [int(num_patches[0]), int(num_patches[1])])),
                nn.Conv2d(
                    in_channels=inp_dim,
                    out_channels=out_dims[0],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_dims[0],
                    out_channels=out_dims[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    bias=True,
                    dilation=1,
                    groups=1,
                ),
            )
        )

        self.reassembles.append(
            nn.Sequential(
                readout_oper[1],
                Transpose(1, 2),
                nn.Unflatten(2, torch.Size(
                    [int(num_patches[0]), int(num_patches[1])])),
                nn.Conv2d(
                    in_channels=inp_dim,
                    out_channels=out_dims[1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_dims[1],
                    out_channels=out_dims[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=True,
                    dilation=1,
                    groups=1,
                ),
            )
        )

        self.reassembles.append(
            nn.Sequential(
                readout_oper[2],
                Transpose(1, 2),
                nn.Unflatten(2, torch.Size(
                    [int(num_patches[0]), int(num_patches[1])])),
                nn.Conv2d(
                    in_channels=inp_dim,
                    out_channels=out_dims[2],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
        )

        self.reassembles.append(
            nn.Sequential(
                readout_oper[3],
                Transpose(1, 2),
                nn.Unflatten(2, torch.Size(
                    [int(num_patches[0]), int(num_patches[1])])),
                nn.Conv2d(
                    in_channels=inp_dim,
                    out_channels=out_dims[3],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.Conv2d(
                    in_channels=out_dims[3],
                    out_channels=out_dims[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            )
        )

    def forward(self, embs):
        results = []

        for emb, reassemble in zip(embs, self.reassembles):
            x = reassemble[0:2](emb)
            if x.ndim == 3:
                x = reassemble[2](x)

            x = reassemble[3:len(reassemble)](x)
            results.append(x)

        return results


class RefineBlock(nn.Module):
    def __init__(self, in_shape, out_shape, activation, groups=1, expand=False, use_bn=False, **kwargs):
        super().__init__()

        out_shape1 = out_shape
        out_shape2 = out_shape
        out_shape3 = out_shape
        out_shape4 = out_shape
        if expand == True:
            out_shape1 = out_shape
            out_shape2 = out_shape * 2
            out_shape3 = out_shape * 4
            out_shape4 = out_shape * 8

        self.layers_rn = nn.ModuleList()

        self.layers_rn.append(
            nn.Conv2d(
                in_shape[0],
                out_shape1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=groups,
            )
        )
        self.layers_rn.append(
            nn.Conv2d(
                in_shape[1],
                out_shape2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=groups,
            )
        )
        self.layers_rn.append(
            nn.Conv2d(
                in_shape[2],
                out_shape3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=groups,
            )
        )
        self.layers_rn.append(
            nn.Conv2d(
                in_shape[3],
                out_shape4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=groups,
            )
        )

        self.refinenets = nn.ModuleList()

        for _ in range(4):
            self.refinenets.append(_make_fusion_block(out_shape, use_bn, activation))

    def forward(self, embs):
        y = None
        results = []

        for x, layer_rn in zip(embs, self.layers_rn):
            results.append(layer_rn(x))

        for result, refinenet in zip(results[::-1], self.refinenets):
            if y is not None:
                y = refinenet(y, result)
            else:
                y = refinenet(result)

        return y


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = F.interpolate
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


class FeatureFusionBlock(nn.Module):
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
        super(FeatureFusionBlock, self).__init__()

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

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

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

        output = F.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
