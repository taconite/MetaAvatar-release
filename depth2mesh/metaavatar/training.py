import os
import numbers
import trimesh
import numpy as np
import torch
from depth2mesh.metaavatar import models
from depth2mesh.training import BaseTrainer

from depth2mesh.utils.loss_functions import sdf as sdf_loss

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device

    '''

    def __init__(self, model, optimizer,
                 stage='skinning_weights',
                 inner_step_size=1e-4,
                 inner_batch_size=1,
                 meta_learner='reptile',
                 no_test_loss=False,
                 n_inner_steps=24,
                 device=None,
                 **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.n_inner_steps = n_inner_steps
        self.inner_step_size = inner_step_size
        self.inner_batch_size = inner_batch_size
        self.meta_learner = meta_learner
        self.no_test_loss = no_test_loss

        if stage not in ['skinning_weights', 'meta', 'meta-hyper']:
            raise ValueError('Unknown stage for MetaAvatar: {}'.format(stage))

        self.stage = stage

        self.decoder = kwargs['decoder']
        self.decoder_kwargs = kwargs['decoder_kwargs']

    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(data, it)
        # We only back-prop here for skinning network training (stage 1).
        # For meta-learning of SIREN (stage 2) and hypernetworks (stage 3),
        # the (meta-)gradients are already computed in compute_meta_loss()
        # function
        if self.stage not in ['meta', 'meta-hyper']:
            loss_dict['total_loss'].backward()

        self.optimizer.step()
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}

    def eval_step(self, data, model_dict=None):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        eval_dict = {}

        points_corr = data.get('points_corr').to(device)
        points_corr_hat = data.get('points_corr.a_pose').to(device)
        batch_size, T, D = points_corr.size()

        loc = data.get('points_corr.loc').to(device)
        trans = data.get('points_corr.trans').to(device)
        coord_min = data.get('points_corr.coord_min').to(device)
        coord_max = data.get('points_corr.coord_max').to(device)
        center = data.get('points_corr.center').to(device)
        coord_min = coord_min.view(batch_size, 1, -1)
        coord_max = coord_max.view(batch_size, 1, -1)
        center = center.view(batch_size, 1, -1)

        bone_transforms = data.get('points_corr.bone_transforms').to(device)
        bone_transforms_02v = data.get('points_corr.bone_transforms_02v').to(device)
        minimal_shape = data.get('points_corr.minimal_shape').to(device)

        inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)

        pose = data.get('points_corr.pose').to(device)
        normals_a_pose = data.get('points_corr.normals_a_pose', None)
        if normals_a_pose is not None:
            normals_a_pose = normals_a_pose.to(device)

        kwargs = {}
        scale = data.get('points_corr.scale').to(device)
        kwargs.update({'scale': scale.view(-1, 1, 1),
                       'loc': loc.view(-1, 1, 3),
                       'p_hat': points_corr_hat,
                       'coord_min': coord_min,
                       'coord_max': coord_max,
                       'center': center,
                       'pose': pose,
                       'normals_a_pose': normals_a_pose,
                       'bone_transforms': bone_transforms,
                       'bone_transforms_02v': bone_transforms_02v,
                       'minimal_shape': minimal_shape})

        eval_dict = {}

        with torch.no_grad():
            # Forward model
            out_dict = self.model(points_corr, inputs, stage=self.stage, **kwargs)

        if self.stage == 'skinning_weights':
            padding = (coord_max - coord_min) * 0.05
            points_corr_hat = (points_corr_hat / 2.0 + 0.5) * 1.1 * (coord_max - coord_min) + coord_min - padding + center
            p_hat = out_dict['p_hat'] * kwargs['scale'] / 1.5

            dist_bwd = torch.norm(points_corr_hat - p_hat, dim=-1).mean().item()

            eval_dict['dist'] = dist_bwd
        elif self.stage in ['meta', 'meta-hyper']:
            if hasattr(self.model.decoder, 'hierarchical_pose'):
                if self.model.decoder.hierarchical_pose and self.stage == 'meta-hyper':
                    rots = data.get('points_corr.rots').to(device)
                    Jtrs = data.get('points_corr.Jtrs').to(device)
                    out_dict.pop('cond', None)
                    out_dict.update({'rots': rots, 'Jtrs': Jtrs})

            loss_meta, loss_dict_meta = self.compute_meta_loss(out_dict, it=None)

            eval_dict['loss'] = loss_meta.item()
        else:
            raise ValueError('Unknown training stage. Supported stages are: skinning_weights, meta, meta-hyper')

        return eval_dict

    def compute_sdf_loss(self, model_output, gt):
        loss_dict = sdf_loss(model_output, gt)
        total_loss = torch.zeros(1, device=self.device)
        for loss_name, loss in loss_dict.items():
            total_loss += loss.mean()

        return total_loss, loss_dict

    def compute_meta_loss(self, in_dict, it):
        outer_loss = torch.zeros(1, device=self.device)
        outer_loss_dict = {}

        coords = in_dict['coords']
        batch_size = coords.size(0)
        sdf = in_dict['sdf']
        normals = in_dict['normals']

        if self.stage == 'meta':
            if 'cond' in in_dict.keys():
                cond = in_dict['cond']
            else:
                cond = None
        else:
            decoder_input = {'coords': coords}
            if 'cond' in in_dict.keys():
                decoder_input.update({'cond': in_dict['cond']})
            if 'rots' in in_dict.keys():
                decoder_input.update({'rots': in_dict['rots']})
            if 'Jtrs' in in_dict.keys():
                decoder_input.update({'Jtrs': in_dict['Jtrs']})

            gt = {'sdf': sdf, 'normals': normals}

        if self.meta_learner == 'maml':
            raise NotImplementedError('MAML is not implemented!')
        elif self.meta_learner == 'reptile':
            if self.stage == 'meta':
                for b_idx in range(batch_size):
                    decoder_clone = models.decoder_dict[self.decoder](**self.decoder_kwargs)
                    decoder_clone.load_state_dict(self.model.decoder.state_dict())
                    decoder_clone = decoder_clone.to(self.device)

                    decoder_input = {'coords': coords[b_idx].unsqueeze(0)}
                    if cond is not None:
                        decoder_input.update({'cond': cond[b_idx].unsqueeze(0)})

                    gt = {'sdf': sdf[b_idx].unsqueeze(0), 'normals': normals[b_idx].unsqueeze(0)}

                    inner_optimizer = torch.optim.Adam(decoder_clone.parameters(), lr=self.inner_step_size)

                    for iter in range(self.n_inner_steps):
                        inner_output = decoder_clone(decoder_input)
                        inner_loss, inner_loss_dict = self.compute_sdf_loss(inner_output, gt)

                        decoder_clone.zero_grad()
                        inner_loss.backward()
                        inner_optimizer.step()

                    for p, target_p in zip(self.model.decoder.parameters(), decoder_clone.parameters()):
                        if p.grad is None:
                            p.grad = torch.zeros(p.size()).cuda()

                        p.grad.data.add_((p.data - target_p.data) / batch_size)

                    outer_loss += inner_loss / batch_size
                    for k, v in inner_loss_dict.items():
                        if k in outer_loss_dict.keys():
                            outer_loss_dict[k] += v / batch_size
                        else:
                            outer_loss_dict[k] = v / batch_size

            elif self.stage == 'meta-hyper':
                # This part does inner-loop in SGD manner
                decoder_clone = models.decoder_dict[self.decoder](**self.decoder_kwargs)
                decoder_clone.load_state_dict(self.model.decoder.state_dict())
                decoder_clone = decoder_clone.to(self.device)

                if hasattr(decoder_clone, 'hierarchical_pose'):
                    if decoder_clone.hierarchical_pose:
                        inner_optimizer = torch.optim.Adam(
                            params = [
                                {
                                    "params": decoder_clone.net.parameters(),
                                    "lr": self.inner_step_size,
                                },
                                {
                                    "params": decoder_clone.pose_encoder.parameters(),
                                    "lr": 1e-4,
                                }
                            ]
                        )
                    else:
                        inner_optimizer = torch.optim.Adam(decoder_clone.parameters(), lr=self.inner_step_size)
                else:
                    inner_optimizer = torch.optim.Adam(decoder_clone.parameters(), lr=self.inner_step_size)

                if self.no_test_loss:
                    train_decoder_input = decoder_input
                    train_gt = gt

                    train_batch_size = batch_size
                else:
                    if batch_size > 1:
                        train_decoder_input = {k: v[:batch_size//2] for k, v in decoder_input.items()}
                        train_gt = {k: v[:batch_size//2] for k, v in gt.items()}

                        train_batch_size = batch_size // 2

                        test_decoder_input = {k: v[batch_size//2:] for k, v in decoder_input.items()}
                        test_gt = {k: v[batch_size//2:] for k, v in gt.items()}

                        test_batch_size = batch_size - batch_size // 2
                    else:
                        train_decoder_input = decoder_input
                        train_gt = gt

                        test_decoder_input = decoder_input
                        test_gt = gt

                        train_batch_size = test_batch_size = 1

                # Train on training set
                for iter in range(self.n_inner_steps):
                    b_inds = np.random.permutation(list(range(train_batch_size)))
                    for start_idx in range(0, train_batch_size, self.inner_batch_size):
                        curr_batch_size = min(self.inner_batch_size, train_batch_size - start_idx)
                        batch_decoder_input = {k: v[b_inds[start_idx:start_idx+curr_batch_size]] for k, v in train_decoder_input.items()}
                        batch_gt = {k: v[b_inds[start_idx:start_idx+curr_batch_size]] for k, v in train_gt.items()}

                        inner_output = decoder_clone(batch_decoder_input)
                        inner_loss, inner_loss_dict = self.compute_sdf_loss(inner_output, batch_gt)

                        if self.stage == 'meta-hyper' and 'params' in inner_output.keys():
                            params = torch.cat(inner_output['params'], dim=1)
                            n_params = params.size(-1)
                            inner_loss += params.norm(dim=-1).mean() * 1e2 / n_params
                            # log_str = 'Iter {}: '.format(iter)
                            # for k, v in inner_loss_dict.items():
                            #     log_str += '{} loss: {:.4f},'.format(k, v)

                            # print (log_str)

                        decoder_clone.zero_grad()
                        inner_loss.backward()
                        inner_optimizer.step()

                        if iter >= self.n_inner_steps - 1:
                            if self.no_test_loss:
                                outer_loss += inner_loss * curr_batch_size / train_batch_size

                            for k, v in inner_loss_dict.items():
                                k = 'train_' + k
                                if k in outer_loss_dict.keys():
                                    outer_loss_dict[k] += v * curr_batch_size / train_batch_size
                                else:
                                    outer_loss_dict[k] = v * curr_batch_size / train_batch_size

                # Generalization loss on test set
                if not self.no_test_loss:
                    for start_idx in range(0, test_batch_size, self.inner_batch_size):
                        curr_batch_size = min(self.inner_batch_size, test_batch_size - start_idx)
                        batch_decoder_input = {k: v[start_idx:start_idx+curr_batch_size] for k, v in test_decoder_input.items()}
                        batch_gt = {k: v[start_idx:start_idx+curr_batch_size] for k, v in test_gt.items()}

                        inner_output = decoder_clone(batch_decoder_input)
                        inner_loss, inner_loss_dict = self.compute_sdf_loss(inner_output, batch_gt)

                        if self.stage == 'meta-hyper' and 'params' in inner_output.keys():
                            params = torch.cat(inner_output['params'], dim=1)
                            n_params = params.size(-1)
                            inner_loss += params.norm(dim=-1).mean() * 1e2 / n_params
                            # log_str = 'Iter {}: '.format(iter)
                            # for k, v in inner_loss_dict.items():
                            #     log_str += '{} loss: {:.4f},'.format(k, v)

                            # print (log_str)

                        decoder_clone.zero_grad()
                        inner_loss.backward()
                        inner_optimizer.step()
                        outer_loss += inner_loss * curr_batch_size / test_batch_size

                        for k, v in inner_loss_dict.items():
                            if k in outer_loss_dict.keys():
                                outer_loss_dict[k] += v * curr_batch_size / test_batch_size
                            else:
                                outer_loss_dict[k] = v * curr_batch_size / test_batch_size

                outer_loss_dict['batch_size'] = batch_size
                # Copy the gradient from the faster model to the meta-model
                for p, target_p in zip(self.model.decoder.parameters(), decoder_clone.parameters()):
                    if p.grad is None:
                        p.grad = torch.zeros(p.size()).cuda()

                    p.grad.data.add_(p.data - target_p.data)
            else:
                raise ValueError('Unsupported stage option. Supported stages are: skinning_weights, meta, meta-hyper')
        else:
            raise ValueError('Unsupported meta-learner {}. Supported meta-learners is reptile'.format(self.meta_learner))

        return outer_loss, outer_loss_dict

    def compute_loss(self, data, it):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        points_corr = data.get('points_corr').to(device)
        points_corr_cano = data.get('points_corr.cano').to(device)
        points_corr_hat = data.get('points_corr.a_pose').to(device)
        bone_transforms = data.get('points_corr.bone_transforms').to(device)
        bone_transforms_02v = data.get('points_corr.bone_transforms_02v').to(device)
        skinning_weights = data.get('points_corr.skinning_weights').to(device)
        minimal_shape = data.get('points_corr.minimal_shape').to(device)

        batch_size = points_corr.size(0)
        loc = data.get('points_corr.loc').to(device)

        pose = data.get('points_corr.pose').to(device)
        normals_a_pose = data.get('points_corr.normals_a_pose', None)
        if normals_a_pose is not None:
            normals_a_pose = normals_a_pose.to(device)

        kwargs = {}
        scale = data.get('points_corr.scale').to(device)
        coord_min = data.get('points_corr.coord_min').to(device)
        coord_max = data.get('points_corr.coord_max').to(device)
        center = data.get('points_corr.center').to(device)

        coord_min = coord_min.view(batch_size, 1, -1)
        coord_max = coord_max.view(batch_size, 1, -1)
        center = center.view(batch_size, 1, -1)

        kwargs.update({'scale': scale.view(-1, 1, 1),
                       'loc': loc.view(-1, 1, 3),
                       'p_hat': points_corr_hat,
                       'coord_min': coord_min,
                       'coord_max': coord_max,
                       'center': center,
                       'pose': pose,
                       'normals_a_pose': normals_a_pose,
                       'bone_transforms': bone_transforms,
                       'bone_transforms_02v': bone_transforms_02v,
                       'minimal_shape': minimal_shape})

        inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)

        loss_dict = {}

        # Forward model
        out_dict = self.model(points_corr, inputs, stage=self.stage, **kwargs)

        # Skinning network loss
        if self.stage in ['skinning_weights']:
            p_rp = out_dict['p_rp']
            pts_W_bwd = out_dict['pts_W_bwd']
            pts_W_fwd = out_dict['pts_W_fwd']
            loss_reproj = torch.norm(p_rp - points_corr, dim=-1).mean()
            loss_dict['loss_reproj'] = loss_reproj
            loss_sem = torch.abs(pts_W_bwd - pts_W_fwd).sum(-1).mean()
            loss_dict['loss_sem'] = loss_sem
            loss_skin = torch.abs(pts_W_bwd - skinning_weights).sum(-1).mean() + torch.abs(pts_W_fwd - skinning_weights).sum(-1).mean()
            loss_dict['loss_skin'] = loss_skin
        else:
            loss_reproj = torch.zeros(1, device=self.device)
            loss_sem = torch.zeros(1, device=self.device)
            loss_skin = torch.zeros(1, device=self.device)

        # Meta-learning loss
        if self.stage in ['meta', 'meta-hyper']:
            if hasattr(self.model.decoder, 'hierarchical_pose'):
                if self.model.decoder.hierarchical_pose and self.stage == 'meta-hyper':
                    rots = data.get('points_corr.rots').to(device)
                    Jtrs = data.get('points_corr.Jtrs').to(device)
                    out_dict.pop('cond', None)
                    out_dict.update({'rots': rots, 'Jtrs': Jtrs})

            loss_meta, loss_dict_meta = self.compute_meta_loss(out_dict, it)
            loss_dict['loss_meta'] = loss_meta
            loss_dict.update(loss_dict_meta)
        else:
            loss_meta = torch.zeros(1, device=self.device)

        # Total weighted sum
        loss = loss_reproj + loss_sem + 10*loss_skin + loss_meta

        loss_dict['total_loss'] = loss

        return loss_dict
