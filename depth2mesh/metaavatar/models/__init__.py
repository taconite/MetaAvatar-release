import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import numbers
# from torch import distributions as dist
from depth2mesh import encoder
from depth2mesh.encoder.pointnet import normalize_coordinate
from depth2mesh.metaavatar.models import (decoder, siren_modules)

from depth2mesh.utils.loss_functions import sdf
from collections import OrderedDict

# Decoder dictionary
decoder_dict = {
    'unstructured_groupnorm': decoder.NASAGroupNormDecoder,
    'single_bvp': siren_modules.SingleBVPNet,
    'hyper_bvp': siren_modules.HyperBVPNet,
}

class MetaAvatar(nn.Module):
    ''' MetaAvatar model class.

    It consists of a decoder and, depending on the respective settings, an
    encoder.

    Args:
        encoder (nn.Module): encoder network
        device (device): PyTorch device
        input_type (str): type of input

    '''

    def __init__(
            self,
            decoder=None,
            encoder_fwd=None,
            encoder_bwd=None,
            skinning_decoder_fwd=None,
            skinning_decoder_bwd=None,
            device=None,
            input_type=None,
            simple_concat=False,
            **kwargs):
        super().__init__()
        self.device = device
        self.input_type = input_type

        self.encoder = encoder
        self.encoder_fwd = encoder_fwd
        self.encoder_bwd = encoder_bwd
        self.decoder = decoder
        self.skinning_decoder_fwd = skinning_decoder_fwd
        self.skinning_decoder_bwd = skinning_decoder_bwd

        self.simple_concat = simple_concat

    def LBS(self, p, pts_W, bone_transforms, bone_transforms_02v, forward=True, normals=None):
        batch_size = p.size(0)
        T = torch.matmul(pts_W, bone_transforms.view(batch_size, -1, 16)).view(batch_size, -1, 4, 4)
        T_v = torch.matmul(pts_W, bone_transforms_02v.view(batch_size, -1, 16)).view(batch_size, -1, 4, 4)
        T = torch.matmul(T, torch.inverse(T_v))
        if not forward:
            T = torch.inverse(T)

        p = F.pad(p, (0, 1), value=1).unsqueeze(-1)
        p_transformed = torch.matmul(T, p)[:, :, :3].squeeze(-1)

        if normals is not None:
            normals_a_pose = torch.matmul(T[:, :, :3, :3], normals.unsqueeze(-1)).squeeze(-1)
        else:
            normals_a_pose = None

        return p_transformed, normals_a_pose

    def forward(self, p, inputs, stage='skinning_weights', **kwargs):
        ''' Makes a forward pass through the network.

        Args:
            p (tensor): points tensor
            inputs (tensor): input tensor
        '''

        if stage not in ['skinning_weights', 'meta', 'meta-hyper']:
            raise ValueError('Unknown stage for MetaAvatar: {}'.format(stage))

        batch_size, T, _ = p.size()
        device = self.device

        out_dict = {}
        if stage == 'skinning_weights':
            # Predict backward skinning weights
            c = self.encode_inputs(inputs, forward=False, **kwargs)
            c_p = self.get_point_features(p, c=c, forward=False, **kwargs)

            pts_W_bwd = self.decode_w(p, c=c_p, forward=False, **kwargs)
            pts_W_bwd = F.softmax(pts_W_bwd, dim=1).transpose(1, 2)

            normals = kwargs.get('normals', None)
            p_hat, normals_a_pose = self.LBS(p, pts_W_bwd, kwargs['bone_transforms'], kwargs['bone_transforms_02v'], forward=False, normals=normals)
            p_hat = p_hat.detach()

            # Normalize input point clouds
            with torch.no_grad():
                p_hat_org = p_hat * kwargs['scale'] / 1.5
                coord_max = p_hat_org.max(dim=1, keepdim=True)[0]
                coord_min = p_hat_org.min(dim=1, keepdim=True)[0]

                total_size = (coord_max - coord_min).max(dim=-1, keepdim=True)[0]
                scale = torch.clamp(total_size, min=1.6)
                loc = (coord_max + coord_min) / 2

                sc_factor = 1.0 / scale * 1.5

                p_hat_norm = (p_hat_org - loc) * sc_factor

                inp_norm = p_hat_norm

            # Predict forward skinning weights
            c = self.encode_inputs(inp_norm, forward=True, **kwargs)
            c_p = self.get_point_features(p_hat_norm, c=c, forward=True, **kwargs)

            pts_W_fwd = self.decode_w(p_hat_norm, c=c_p, forward=True, **kwargs)
            pts_W_fwd = F.softmax(pts_W_fwd, dim=1).transpose(1, 2)

            p_rp, _ = self.LBS(p_hat, pts_W_fwd, kwargs['bone_transforms'], kwargs['bone_transforms_02v'], forward=True)

            out_dict.update({'p_rp': p_rp, 'p_hat': p_hat, 'normals_a_pose': normals_a_pose, 'pts_W_bwd': pts_W_bwd, 'pts_W_fwd': pts_W_fwd})

        if stage in ['meta', 'meta-hyper']:
            # We train the SDF decoder with:
            # 1) GT points (with normals) in the A-pose
            # 2) GT SMPL poses (in the form of quaternions)
            # via meta-learning
            # p_hat = p_hat.clone().detach()  # we don't want to propagate gradient through p_hat
            p_hat = kwargs.get('p_hat')
            pose = kwargs.get('pose')

            normals_a_pose = kwargs.get('normals_a_pose')

            on_surface_samples = p_hat.size(1) # // 2 # dirty hack to reduce memory usage of MAML algorithm
            off_surface_samples = on_surface_samples
            total_samples = on_surface_samples + off_surface_samples

            coords = p_hat

            on_surface_coords = coords
            on_surface_normals = normals_a_pose

            off_surface_coords = (torch.rand(batch_size, off_surface_samples, 3, device=device, dtype=torch.float32) - 0.5) * 2
            off_surface_normals = torch.ones(batch_size, off_surface_samples, 3, device=device, dtype=torch.float32) * -1

            sdf = torch.zeros(batch_size, total_samples, 1, device=device, dtype=torch.float32)  # on-surface = 0
            sdf[:, on_surface_samples:, :] = -1  # off-surface = -1

            coords_in = torch.cat([on_surface_coords, off_surface_coords], dim=1)
            normals_in = torch.cat([on_surface_normals, off_surface_normals], dim=1)
            poses_in = pose

            if stage == 'meta-hyper' and not self.simple_concat:
                out_dict.update({'cond': poses_in})
            elif self.simple_concat:
                out_dict.update({'cond': poses_in.unsqueeze(1).repeat(1, total_samples, 1)})

            out_dict.update({'coords': coords_in})
            out_dict.update({'sdf': sdf, 'normals': normals_in})

        return out_dict

    def encode_inputs(self, inputs, forward=True, **kwargs):
        ''' Returns the encoding.

        Args:
            inputs (tensor): input tensor)
        '''
        batch_size = inputs.shape[0]
        device = self.device

        if forward and self.encoder_fwd is not None:
            c = self.encoder_fwd(inputs, **kwargs)
        elif not forward and self.encoder_bwd is not None:
            c = self.encoder_bwd(inputs, **kwargs)
        else:
            c = torch.empty(batch_size, 0).to(device)

        return c

    def get_point_features(self, p, c=None, forward=True, **kwargs):
        ''' Returns point-aligned features for points on a convolutional feature map.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        if isinstance(c, OrderedDict):
            normalized_scale = self.encoder_fwd.normalized_scale if forward else self.encoder_bwd.normalized_scale
            padding = self.encoder_fwd.padding if forward else self.encoder_bwd.padding

            point_features = []
            if normalized_scale:
                scale = 1.0
            else:
                scale = kwargs['scale']

            for k, v in c.items():
                if k in ['xz', 'xy', 'yz']:
                    projected = normalize_coordinate(p.clone(), scale, plane=k, padding=padding) # normalize to the range of (0, 1)
                    projected = (projected * 2 - 1).unsqueeze(2)    # grid_sample accepts inputs in range [-1, 1]
                    point_features.append(nn.functional.grid_sample(v, projected, align_corners=True).squeeze(-1))
                elif k in ['grid']:
                    raise NotImplementedError('Convolutional feature {} not implemented yet'.format(k))
                else:
                    raise ValueError('Wrong type of convolutional feature: {}.'.format(k))

            return torch.cat(point_features, dim=1)    # B x c_dim x T
        else:
            raise ValueError('Input c is expected to be an OrderedDict, but got {}'.format(type(c)))

    def decode_w(self, p, c=None, forward=True, **kwargs):
        ''' Returns skinning weights for the points p.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        pts_W = self.skinning_decoder_fwd(p, c=c, **kwargs) if forward else self.skinning_decoder_bwd(p, c=c, **kwargs)

        return pts_W

    def decode(self, model_input, params=None):
        ''' run the (meta) decoder.
        Args:
            model_input (tensor): input to the decoder
            time_val (tensor): time values
            inputs (tensor): input tensor
        '''
        return self.decoder(model_input, params=params)

    def decoder_zero_grad(self):
        ''' run the (meta) decoder.
        Args:
            model_input (tensor): input to the decoder
            time_val (tensor): time values
            inputs (tensor): input tensor
        '''
        return self.decoder.zero_grad()

    def get_decoder_params(self):
        ''' Returns decoder parameters. '''
        return OrderedDict(self.decoder.meta_named_parameters())
