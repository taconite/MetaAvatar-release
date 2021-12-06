'''From the SIREN repository https://github.com/vsitzmann/siren
'''

import torch
import torch.nn.functional as F

import depth2mesh.utils.diff_operators as diff_operators

def sdf(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt.get('sdf', None)
    gt_normals = gt.get('normals', None)
    gt_labels = gt.get('part_labels', None)

    if gt_sdf is not None and gt_normals is not None:
        coords = model_output['model_in']
        pred_sdf = model_output['model_out']
        pred_part_sdfs = model_output.get('part_sdfs')

        gradient = diff_operators.gradient(pred_sdf, coords)

        # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
        sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
        inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
        normal_constraint = torch.where((gt_sdf != -1) & (gt_normals.sum(-1) != 0).unsqueeze(-1), 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                        torch.zeros_like(gradient[..., :1]))
        grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
        if gt_labels is not None and pred_part_sdfs is not None:
            device = gt_labels.device
            gt_labels = gt_labels.unsqueeze(-1)
            pred_part_sdfs = model_output['part_sdfs']

            num_joints = pred_part_sdfs.size(-1)

            p_targets = torch.zeros(gt_labels.size(0), gt_labels.size(1), num_joints+1, device=device)
            p_targets.scatter_(2, gt_labels, 1)    # One-hot vector for each point B x n_pts x num_joints
            no_label = (p_targets[:, :, -1] == 1).view(gt_labels.size(0), gt_labels.size(1), 1).repeat(1, 1, num_joints)
            p_targets = p_targets[:, :, :-1]

            on_surface_samples = pred_part_sdfs.size(1) // 2
            pred_part_sdfs = pred_part_sdfs[:, :on_surface_samples, :]

            part_sdf_constraint = torch.where((p_targets == 1) & ~no_label, pred_part_sdfs, torch.zeros_like(pred_part_sdfs)).sum(-1)
            part_inter_constraint = torch.where((p_targets == 1) | no_label, torch.zeros_like(pred_part_sdfs), torch.exp(-1e2 * pred_part_sdfs)).sum(-1)
            # Exp      # Lapl
            # -----------------
            return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
                    'part_sdfs': torch.abs(part_sdf_constraint).mean() * 3e2,
                    'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
                    'part_inter': part_inter_constraint.mean() * 1e1,  # 1e2                   # 1e3
                    'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
                    # 'chamfer': chamfer_constraint.mean() *  1e2,
                    'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1
        else:
            # Exp      # Lapl
            # -----------------
            return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
                    'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
                    'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
                    # 'chamfer': chamfer_constraint.mean() *  1e2,
                    'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1
    else:
        pred_sdf_backward = model_output['model_out_backward']
        pred_sdf_forward = model_output['model_out_forward']

        # Exp      # Lapl
        # -----------------
        return {'sdf_backward': torch.abs(pred_sdf_backward).mean() * 3e3,
                'sdf_forward': torch.abs(pred_sdf_forward).mean() * 3e3}

def sdf_with_mask(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt.get('sdf', None)
    gt_normals = gt.get('normals', None)
    gt_labels = gt.get('part_labels', None)
    mask = gt.get('mask', None)
    div = mask.float().sum()

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where((gt_sdf != -1) & mask, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where((gt_sdf != -1) & mask, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where((gt_sdf != -1) & (gt_normals.sum(-1) != 0).unsqueeze(-1) & mask,
                                    1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).sum() / div * 3e3,  # 1e4      # 3e3
            'inter': inter_constraint.sum() / div * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.sum() / div * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1
