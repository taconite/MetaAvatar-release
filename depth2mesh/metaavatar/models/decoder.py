import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from depth2mesh import hyperlayers
from depth2mesh.layers import (
    GroupNorm1d,
    ResnetBlockGroupNormConv1d,
    ResnetBlockGroupNormShallowConv1d
)

class NASAGroupNormDecoder(nn.Module):
    ''' Unstructured Decoder.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, out_dim=1, c_dim=128,
                 hidden_size=256, gn_groups=24,
		 dropout_prob=0.0, leaky=False):
        super().__init__()
        self.dim = dim
        self.c_dim = c_dim

        # Submodules
        self.fc_p = nn.Conv1d(dim + c_dim, hidden_size, 1, groups=1)
        self.block0 = ResnetBlockGroupNormConv1d(hidden_size, 1, gn_groups, size_out=hidden_size // 2, dropout_prob=dropout_prob, leaky=leaky)
        hidden_size = hidden_size // 2
        self.block1 = ResnetBlockGroupNormConv1d(hidden_size, 1, gn_groups, size_out=hidden_size // 2, dropout_prob=dropout_prob, leaky=leaky)
        hidden_size = hidden_size // 2
        self.block2 = ResnetBlockGroupNormConv1d(hidden_size, 1, gn_groups, size_out=hidden_size // 2, dropout_prob=dropout_prob, leaky=leaky)
        hidden_size = hidden_size // 2
        self.block3 = ResnetBlockGroupNormConv1d(hidden_size, 1, gn_groups, dropout_prob=dropout_prob, leaky=leaky)

        if dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob, inplace=True)
        else:
            self.dropout = None

        self.fc_out = nn.Conv1d(hidden_size, out_dim, 1, groups=1)

        self.gn = GroupNorm1d(gn_groups, hidden_size)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.1)

    def forward(self, p, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # p = p[:, D - self.dim:, :] * kwargs['z_scale']
        if len(c.shape) < 3 and self.c_dim > 0:
            c = c.unsqueeze(2).repeat(1, 1, T)

        net = self.fc_p(torch.cat([p, c], dim=1)) if self.c_dim > 0 else self.fc_p(p)

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)

        if self.dropout is not None:
            out = self.fc_out(self.dropout(self.actvn(self.gn(net))))  # B x 1 x 2048
        else:
            out = self.fc_out(self.actvn(self.gn(net)))  # B x 1 x 2048

        return out.squeeze(1)


class NASAGroupNormShallowDecoder(nn.Module):
    ''' Shallow Unstructured Decoder.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, gn_groups=24,
		 dropout_prob=0.0, leaky=False):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        self.fc_p = nn.Conv1d(dim + c_dim, hidden_size, 1, groups=1)
        self.block0 = ResnetBlockGroupNormShallowConv1d(hidden_size, 1, gn_groups, dropout_prob=dropout_prob, leaky=leaky)
        self.block1 = ResnetBlockGroupNormShallowConv1d(hidden_size, 1, gn_groups, dropout_prob=dropout_prob, leaky=leaky)
        self.block2 = ResnetBlockGroupNormShallowConv1d(hidden_size, 1, gn_groups, dropout_prob=dropout_prob, leaky=leaky)
        self.block3 = ResnetBlockGroupNormShallowConv1d(hidden_size, 1, gn_groups, dropout_prob=dropout_prob, leaky=leaky)

        if dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob, inplace=True)
        else:
            self.dropout = None

        self.fc_out = nn.Conv1d(hidden_size, 1, 1, groups=1)

        self.gn = GroupNorm1d(gn_groups, hidden_size)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.1)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        p = p[:, D - self.dim:, :] * kwargs['z_scale']
        if len(c.shape) < 3:
            c = c.unsqueeze(2).repeat(1, 1, T)

        net = self.fc_p(torch.cat([p, c], dim=1))

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)

        if self.dropout is not None:
            out = self.fc_out(self.dropout(self.actvn(self.gn(net))))  # B x 1 x 2048
        else:
            out = self.fc_out(self.actvn(self.gn(net)))  # B x 1 x 2048

        return out.squeeze(1)
