import torch
import numpy as np
from torch import nn
import os
from depth2mesh.encoder import encoder_dict
from depth2mesh.metaavatar import models, training
from depth2mesh import data
from depth2mesh import config


def get_sdf_decoder(cfg, device):
    ''' Returns a SDF decoder instance.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
    '''
    decoder = cfg['model']['decoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']

    if decoder:
        decoder = models.decoder_dict[decoder](**decoder_kwargs).to(device)
    else:
        decoder = None

    return decoder

def get_skinning_decoder(cfg, device, dim=3, c_dim=0):
    ''' Returns skinning decoder instances, including forward and backward decoders.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        dim (int): points dimension
        c_dim (int): input feature dimension
    '''
    decoder = cfg['model']['skinning_decoder']
    decoder_kwargs = cfg['model']['skinning_decoder_kwargs']

    if decoder is not None:
        decoder_fwd = models.decoder_dict[decoder](
            dim=dim, out_dim=24, c_dim=c_dim,
            **decoder_kwargs).to(device)

        decoder_bwd = models.decoder_dict[decoder](
            dim=dim, out_dim=24, c_dim=c_dim,
            **decoder_kwargs).to(device)

    else:
        decoder_fwd = decoder_bwd = None

    return decoder_fwd, decoder_bwd


def get_skinning_encoder(cfg, device, c_dim=0):
    ''' Returns skinning encoder instances, including forward and backward encoders.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        c_dim (int): output feature dimension
    '''
    encoder = cfg['model']['encoder']
    encoder_kwargs = cfg['model']['encoder_kwargs']

    encoder_kwargs.update({'normalized_scale': cfg['data']['normalized_scale']})

    if encoder is not None:
        encoder_fwd = encoder_dict[encoder](
            c_dim=c_dim // 3 if encoder in ['pointnet_conv'] else c_dim, **encoder_kwargs).to(device)
        encoder_bwd = encoder_dict[encoder](
            c_dim=c_dim // 3 if encoder in ['pointnet_conv'] else c_dim, **encoder_kwargs).to(device)
    else:
        encoder_fwd = encoder_bwd = None

    return encoder_fwd, encoder_bwd


def get_model(cfg, device=None, **kwargs):
    ''' Return the Unstructured Network model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    input_type = cfg['data']['input_type']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    simple_concat = cfg['model']['simple_concat']

    decoder = get_sdf_decoder(cfg, device)
    skinning_decoder_fwd, skinning_decoder_bwd = get_skinning_decoder(cfg, device, dim, c_dim)
    encoder_fwd, encoder_bwd = get_skinning_encoder(cfg, device, c_dim)

    # Get full MetaAvatar model
    model = models.MetaAvatar(
        decoder=decoder,
        encoder_fwd=encoder_fwd,
        encoder_bwd=encoder_bwd,
        skinning_decoder_fwd=skinning_decoder_fwd,
        skinning_decoder_bwd=skinning_decoder_bwd,
        device=device, input_type=input_type,
        simple_concat=simple_concat
        )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the MetaAvatar model
        optimizer (optim.Optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    batch_size = cfg['training']['batch_size']
    model_type = cfg['model']['decoder']

    inner_batch_size = cfg['training']['inner_batch_size']
    meta_learner = cfg['training']['meta_learner']
    no_test_loss = cfg['training']['no_test_loss']
    optim_iterations = cfg['training']['optim_iterations']

    max_operator = cfg['training']['max_operator']

    stage = cfg['training']['stage']
    inner_lr = cfg['training']['inner_lr']

    decoder = cfg['model']['decoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']

    trainer = training.Trainer(
        model, optimizer,
        stage=stage,
        inner_step_size=inner_lr,
        inner_batch_size=inner_batch_size,
        meta_learner=meta_learner,
        no_test_loss=no_test_loss,
        n_inner_steps=optim_iterations,
        device=device,
        decoder=decoder,
        decoder_kwargs=decoder_kwargs,
    )

    return trainer
