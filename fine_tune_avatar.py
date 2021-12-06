import os
import torch
import trimesh
import argparse
import time
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from depth2mesh import config
from depth2mesh.checkpoints import CheckpointIO
from depth2mesh.metaavatar import models

from depth2mesh.utils.logs import create_logger

parser = argparse.ArgumentParser(
    description='Do fine-tuning on validation set, then extract meshes on novel poses.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite already generated results')
parser.add_argument('--subsampling-rate', type=int, default=1,
                    help='subsampling rate for sampling training sequences')
parser.add_argument('--test-start-offset', type=int, default=0,
                    help='start offset testing sequences')
parser.add_argument('--test-subsampling-rate', type=int, default=1,
                    help='subsampling rate for sampling testing sequences')
parser.add_argument('--epochs-per-run', type=int, default=-1,
                    help='Number of epochs to train before restart.')
parser.add_argument('--optim-epochs', type=int, default=-1,
                    help='Number of total epochs  to train.')
parser.add_argument('--num-workers', type=int, default=8,
                    help='Number of workers to use for train and val loaders.')
parser.add_argument('--interpolation', action='store_true', help='Interpolation task.')
parser.add_argument('--high-res', action='store_true', help='Run marching cubes at high resolution (512^3).')
# parser.add_argument('--canonical', action='store_true', help='Extract canonical meshes only (in the original canonical space).')

parser.add_argument('--subject-idx', type=int, default=-1,
                    help='Which subject in the validation set to train (and optionally test)')
parser.add_argument('--test-subject-idx', type=int, default=-1,
                    help='Which subject in the validation set to test. By default it is the same subject for train.')
parser.add_argument('--train-cloth-split', type=str, metavar='LIST', required=True,
                    help='Which cloth-types in the validation set to train on')
parser.add_argument('--train-act-split', type=str, metavar='LIST', required=True,
                    help='Which actions in the validation set to train on')
parser.add_argument('--test-cloth-split', type=str, metavar='LIST', default='',
                    help='Which cloth-types in the validation set to train on')
parser.add_argument('--test-act-split', type=str, metavar='LIST', required=True,
                    help='Which actions in the validation set to train on')
parser.add_argument('--exp-suffix', type=str, default='',
                    help='User defined suffix to distinguish different test runs.')

def get_skinning_weights(pts, src, ref_W):
    """
    Finds skinning weights of pts on src via barycentric interpolation.
    """
    closest_face, closest_points = src.closest_faces_and_points(pts)
    vert_ids, bary_coords = src.barycentric_coordinates_for_points(closest_points, closest_face.astype('int32'))
    pts_W = (ref_W[vert_ids] * bary_coords[..., np.newaxis]).sum(axis=1)

    return pts_W

def compute_sdf_loss(model_output, gt):
    loss_dict = sdf_loss(model_output, gt)
    total_loss = torch.zeros(1, device=device)
    for loss_name, loss in loss_dict.items():
        total_loss += loss.mean()

    return total_loss, loss_dict

def mask_by_reproj_dist(p, p_rp, mode='mean', value=-1):
    if mode == 'mean':
        thr = torch.norm(p - p_rp, dim=-1).mean(-1, keepdim=True)
    else:
        thr = value

    mask = (torch.norm(p - p_rp, dim=-1) < thr).unsqueeze(-1)

    return mask

def normalize_canonical_points(pts, coord_min, coord_max, center):
    pts -= center
    padding = (coord_max - coord_min) * 0.05
    pts = (pts - coord_min + padding) / (coord_max - coord_min) / 1.1
    pts -= 0.5
    pts *= 2.

    return pts

def get_transforms_02v(Jtr):
    from scipy.spatial.transform import Rotation as R
    rot45p = R.from_euler('z', 45, degrees=True).as_matrix()
    rot45n = R.from_euler('z', -45, degrees=True).as_matrix()
    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))
    # Jtr *= sc_factor

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    chain = [1, 4, 7, 10]
    rot = rot45p.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    chain = [2, 5, 8, 11]
    rot = rot45n.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)

    return bone_transforms_02v

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    out_dir = cfg['training']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
    generation_dir += args.exp_suffix
    out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
    out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')
    stage = cfg['training']['stage']
    inner_lr = cfg['training']['inner_lr']

    batch_size = cfg['training']['inner_batch_size']
    input_type = cfg['data']['input_type']
    vis_n_outputs = cfg['generation']['vis_n_outputs']
    if vis_n_outputs is None:
        vis_n_outputs = -1

    train_cloth_split = [v for v in args.train_cloth_split.split(',')]
    test_cloth_split = [v for v in args.test_cloth_split.split(',')] if len(args.test_cloth_split) > 0 else train_cloth_split
    train_act_split = [v for v in args.train_act_split.split(',')]
    test_act_split = [v for v in args.test_act_split.split(',')]

    logger, _ = create_logger(generation_dir, phase='test_subj{}_cloth-{}'.format(args.subject_idx, train_cloth_split[0]), create_tf_logs=False)

    logger.info('Dataset path: {}'.format(cfg['data']['path']))

    single_view = cfg['data']['single_view']
    dataset_name = cfg['data']['dataset']

    train_dataset = config.get_dataset('test', cfg, subject_idx=args.subject_idx, cloth_split=train_cloth_split, act_split=train_act_split, subsampling_rate=args.subsampling_rate)

    cfg['data']['single_view'] = False  # for novel pose synthesis we always have access to full-body mesh in canonical pose
    cfg['data']['use_raw_scans'] = False
    if args.interpolation:
        cfg['data']['path'] = 'data/CAPE_test_sampling-rate-1'
    # else:
    #     args.test_subsampling_rate = 1
    #     args.test_start_offset = 0

    test_dataset = config.get_dataset('test', cfg, subject_idx=args.subject_idx if args.test_subject_idx < 0 else args.test_subject_idx, cloth_split=test_cloth_split, act_split=test_act_split, subsampling_rate=args.test_subsampling_rate, start_offset=args.test_start_offset)

    # Loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=0, shuffle=False)

    # Model
    model = config.get_model(cfg, device=device, dataset=train_dataset)
    ckpt = torch.load(os.path.join(out_dir, cfg['test']['model_file']))
    decoder_state_dict = OrderedDict()

    # Load meta-learned SDF decoder
    for k, v in ckpt['model'].items():
        if k.startswith('module'):
            k = k[7:]

        if k.startswith('decoder'):
            decoder_state_dict[k[8:]] = v

    model.decoder.load_state_dict(decoder_state_dict)

    # Load forward and backward skinning networks, for fine-tuning
    optim_skinning_net_path = cfg['model']['skinning_net1']
    ckpt = torch.load(optim_skinning_net_path)

    encoder_fwd_state_dict = OrderedDict()
    skinning_decoder_fwd_state_dict = OrderedDict()
    encoder_bwd_state_dict = OrderedDict()
    skinning_decoder_bwd_state_dict = OrderedDict()
    for k, v in ckpt['model'].items():
        if k.startswith('module'):
            k = k[7:]

        if k.startswith('skinning_decoder_fwd'):
            skinning_decoder_fwd_state_dict[k[21:]] = v
        elif k.startswith('skinning_decoder_bwd'):
            skinning_decoder_bwd_state_dict[k[21:]] = v
        elif k.startswith('encoder_fwd'):
            encoder_fwd_state_dict[k[12:]] = v
        elif k.startswith('encoder_bwd'):
            encoder_bwd_state_dict[k[12:]] = v

    model.encoder_fwd.load_state_dict(encoder_fwd_state_dict)
    model.encoder_bwd.load_state_dict(encoder_bwd_state_dict)
    model.skinning_decoder_fwd.load_state_dict(skinning_decoder_fwd_state_dict)
    model.skinning_decoder_bwd.load_state_dict(skinning_decoder_bwd_state_dict)

    model.eval()
    import depth2mesh.utils.sdf_meshing as sdf_meshing
    from depth2mesh.utils.loss_functions import sdf_with_mask as sdf_loss

    # Create a clone of meta-learned SDF decoder
    decoder = cfg['model']['decoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    decoder_clone = models.decoder_dict[decoder](**decoder_kwargs)
    decoder_clone.load_state_dict(model.decoder.state_dict())
    decoder_clone = decoder_clone.to(device)

    if stage == 'meta-hyper' and cfg['model']['decoder'] == 'hyper_bvp':
        if model.decoder.hierarchical_pose:
            inner_optimizer = torch.optim.Adam(
                params = [
                    {
                        "params": decoder_clone.net.parameters(),
                        "lr": inner_lr,
                    },
                    {
                        "params": decoder_clone.pose_encoder.parameters(),
                        "lr": 1e-4,
                    }
                ]
            )
        else:
            inner_optimizer = torch.optim.Adam(decoder_clone.parameters(), lr=inner_lr)
    else:
        raise ValueError('Fine-tuning only supports meta-hyper stage \
                          with SDF decoder type hyper_bvp. Got stage {} and SDF \
                          decoder {}'.format(stage, cfg['model']['decoder']))

    # Checkpoint for fine-tuned SDF decoder
    test_optim_ckpt_io = CheckpointIO(generation_dir, model=decoder_clone, optimizer=inner_optimizer)
    test_optim_ckpt_filename = 'test_time_optim_subj{}_cloth-{}.pt'.format(args.subject_idx, train_cloth_split[0])
    logger.info(test_optim_ckpt_filename)
    try:
        load_dict = test_optim_ckpt_io.load(test_optim_ckpt_filename)
    except FileExistsError:
        load_dict = dict()

    epoch_it = load_dict.get('epoch_it', -1)
    proj_thr = cfg['training']['proj_thr']  # re-projection threshold to filter out invalid points mapped by backward LBS

    if args.optim_epochs > 0:
        max_epoch = args.optim_epochs
    else:
        max_epoch = cfg['test']['optim_iterations']

    # Load minimal shape of the target subject, in order to compute bone transformations later
    model_dict = train_dataset.get_model_dict(0)
    subject = model_dict['subject']
    gender = model_dict['gender']
    minimal_shape_path = os.path.join(train_dataset.cape_path, 'cape_release', 'minimal_body_shape', subject, subject + '_minimal.npy')

    if not os.path.exists(minimal_shape_path):
        raise ValueError('Unsupported CAPE subject: {}'.format(subject))

    minimal_shape = np.load(minimal_shape_path)
    bm_path = os.path.join('./body_models/smpl', gender, 'model.pkl')
    from human_body_prior.body_model.body_model import BodyModel
    bm = BodyModel(bm_path=bm_path, num_betas=10, batch_size=1, v_template=minimal_shape).cuda()

    # Time statistics
    time_dict = OrderedDict()
    time_dict['network_time'] = 0

    # Fine-tuning loop
    epoch_cnt = 0
    epochs_to_run = args.epochs_per_run if args.epochs_per_run > 0 else (max_epoch + 1)
    for _ in range(epochs_to_run):
        epoch_it += 1
        if epoch_it >= max_epoch:
            break

        for idx, data in enumerate(train_loader):
            inputs = data.get('inputs').to(device)
            points_corr = data.get('points_corr').to(device)
            poses = data.get('points_corr.pose').to(device)

            scale = data.get('points_corr.scale').to(device)
            scale = scale.view(-1, 1, 1)
            bone_transforms = data.get('points_corr.bone_transforms').to(device)
            bone_transforms_02v = data.get('points_corr.bone_transforms_02v').to(device)
            minimal_shape = data.get('points_corr.minimal_shape').to(device)
            kwargs = {'scale': scale, 'bone_transforms': bone_transforms, 'bone_transforms_02v': bone_transforms_02v, 'minimal_shape': minimal_shape}

            # TODO: we should get rid of this by re-calculating center by bounding volume
            # not mean of points
            coord_min = data.get('points_corr.coord_min').to(device).view(-1, 1, 1)
            coord_max = data.get('points_corr.coord_max').to(device).view(-1, 1, 1)
            center = data.get('points_corr.center').to(device).unsqueeze(1)

            # Use the learned skinning net to transform points to A-pose
            t = time.time()
            with torch.no_grad():
                out_dict = model(inputs, points_corr, stage='skinning_weights', **kwargs)

            points_corr_hat = out_dict.get('p_hat')
            points_corr_reproj = out_dict.get('p_rp')
            normals_a_pose = out_dict.get('normals_a_pose')
            # Do the following:
            # 1) Filter out points whose re-projection distance is greater than the specified threshold
            # 2) Normalize valid points to [-1, 1]^3 for SDF decoder
            mask = mask_by_reproj_dist(points_corr, points_corr_reproj, mode='constant', value=proj_thr)

            points_corr_hat = points_corr_hat * scale / 1.5
            points_corr_hat = normalize_canonical_points(points_corr_hat, coord_min=coord_min, coord_max=coord_max, center=center)

            batch_size = points_corr_hat.size(0)

            # Generate point samples for fine-tuning
            on_surface_samples = points_corr_hat.size(1)
            off_surface_samples = on_surface_samples
            total_samples = on_surface_samples + off_surface_samples

            on_surface_coords = points_corr_hat
            on_surface_normals = normals_a_pose

            off_surface_coords = (torch.rand(batch_size, off_surface_samples, 3, device=device, dtype=torch.float32) - 0.5) * 2
            off_surface_normals = torch.ones(batch_size, off_surface_samples, 3, device=device, dtype=torch.float32) * -1

            sdf = torch.zeros(batch_size, total_samples, 1, device=device, dtype=torch.float32)  # on-surface = 0
            sdf[:, on_surface_samples:, :] = -1  # off-surface = -1

            coords_in = torch.cat([on_surface_coords, off_surface_coords], dim=1)
            mask = torch.cat([mask, torch.ones_like(mask)], dim=1)

            # Use normal information if available.
            if on_surface_normals is not None:
                normals_in = torch.cat([on_surface_normals, off_surface_normals], dim=1)
            else:
                normals_in = torch.zeros_like(coords_in)

            decoder_input = {'coords': coords_in}
            if decoder_clone.hierarchical_pose:
                rots = data.get('points_corr.rots').to(device)
                Jtrs = data.get('points_corr.Jtrs').to(device)
                decoder_input.update({'rots': rots, 'Jtrs': Jtrs})
            else:
                decoder_input.update({'cond': poses})

            gt = {'sdf': sdf, 'normals': normals_in, 'mask': mask}

            # Forward pass and compute loss
            inner_output = decoder_clone(decoder_input)
            inner_loss, inner_loss_dict = compute_sdf_loss(inner_output, gt)

            # Regularize on predicted SDF parameters
            params = torch.cat(inner_output['params'], dim=1)
            n_params = params.size(-1)
            inner_loss += params.norm(dim=-1).mean() * 1e2 / n_params

            # Do one step of optimization
            decoder_clone.zero_grad()
            inner_loss.backward()
            inner_optimizer.step()

            # Update timing
            time_dict['network_time'] += time.time() - t

            # Logging
            log_str = 'Epoch {}: '.format(epoch_it)
            for k, v in inner_loss_dict.items():
                log_str += '{} loss: {:.4f},'.format(k, v.item())

            logger.info(log_str)

        epoch_cnt += 1

    logger.info('Elapsed network time: {} seconds.'.format(time_dict['network_time']))

    # Save fine-tuned model
    if epoch_cnt > 0:
        test_optim_ckpt_io.save(test_optim_ckpt_filename, epoch_it=epoch_it)

    # If we have not reached desired fine-tuning epoch, then exit with code 3.
    # This for job-chaining on HPC clusters. You can ignore this if you run
    # fine-tuning on local machines.
    if epoch_it < max_epoch:
        exit(3)

    # Novel pose synthesis
    model_count = 0
    faces = np.load('body_models/misc/faces.npz')['faces']
    all_skinning_weights = dict(np.load('body_models/misc/skinning_weights_all.npz'))

    # Load forward and backward skinning networks, for novel-pose synthesis
    optim_skinning_net_path = cfg['model']['skinning_net2']
    ckpt = torch.load(optim_skinning_net_path)

    encoder_fwd_state_dict = OrderedDict()
    skinning_decoder_fwd_state_dict = OrderedDict()
    encoder_bwd_state_dict = OrderedDict()
    skinning_decoder_bwd_state_dict = OrderedDict()
    for k, v in ckpt['model'].items():
        if k.startswith('module'):
            k = k[7:]

        if k.startswith('skinning_decoder_fwd'):
            skinning_decoder_fwd_state_dict[k[21:]] = v
        elif k.startswith('skinning_decoder_bwd'):
            skinning_decoder_bwd_state_dict[k[21:]] = v
        elif k.startswith('encoder_fwd'):
            encoder_fwd_state_dict[k[12:]] = v
        elif k.startswith('encoder_bwd'):
            encoder_bwd_state_dict[k[12:]] = v

    model.encoder_fwd.load_state_dict(encoder_fwd_state_dict)
    model.encoder_bwd.load_state_dict(encoder_bwd_state_dict)
    model.skinning_decoder_fwd.load_state_dict(skinning_decoder_fwd_state_dict)
    model.skinning_decoder_bwd.load_state_dict(skinning_decoder_bwd_state_dict)

    # Indices of joints for which we set their rotations to 0
    zero_indices = np.array([10, 11, 22, 23])   # feet and hands
    zero_indices_parents = [7, 8, 20, 21]   # and their parents

    # Novel-pose synthesis over test data
    for _, data in enumerate(test_loader):
        model_count += 1

        # Output folders
        cloth_dir = os.path.join(generation_dir, 'cloth')

        # Get index etc.
        idx = data['idx'].item()

        model_dict = test_dataset.get_model_dict(idx)

        if input_type == 'pointcloud':
            subset = model_dict['subset']
            subject = model_dict['subject']
            sequence = model_dict['sequence']
            gender = model_dict['gender']
            data_path = model_dict['data_path']
            filebase = os.path.basename(data_path)[:-4]
        else:
            raise ValueError('Unknown input type: {}'.format(input_type))

        folder_name = os.path.join(subset, subject, sequence)

        cloth_dir = os.path.join(cloth_dir, folder_name)

        if not os.path.exists(cloth_dir):
            os.makedirs(cloth_dir)

        poses = data.get('points_corr.pose').to(device)
        minimal_shape = data.get('points_corr.minimal_shape').to(device)

        colors = np.load('body_models/misc/part_colors.npz')['colors']

        if args.high_res:
            cano_filename = os.path.join(cloth_dir, filebase + '.cano.high')
            posed_filename = os.path.join(cloth_dir, filebase + '.posed.high')
        else:
            cano_filename = os.path.join(cloth_dir, filebase + '.cano')
            posed_filename = os.path.join(cloth_dir, filebase + '.posed')

        rots = data.get('points_corr.rots').to(device)
        Jtrs = data.get('points_corr.Jtrs').to(device)

        # Run grid evaluation and marching-cubes to obtain mesh in canonical space
        if hasattr(decoder_clone, 'hierarchical_pose'):
            if decoder_clone.hierarchical_pose:
                sdf_meshing.create_mesh(decoder_clone,
                                        thetas={'rots': rots, 'Jtrs': Jtrs},
                                        filename=cano_filename, N=512 if args.high_res else 256,
                                        max_batch=64 ** 3)
            else:
                sdf_meshing.create_mesh(decoder_clone,
                                        thetas=poses[0],
                                        filename=cano_filename, N=512 if args.high_res else 256,
                                        max_batch=64 ** 3)
        else:
            sdf_meshing.create_mesh(decoder_clone,
                                    thetas=poses,
                                    filename=cano_filename, N=512 if args.high_res else 256,
                                    max_batch=64 ** 3)

        # Convert canonical pose shape from the its normalized space to pointcloud encoder space
        a_pose_trimesh = trimesh.load(cano_filename + '.ply', process=False)

        # Filter out potential floating blobs
        labels = trimesh.graph.connected_component_labels(a_pose_trimesh.face_adjacency)
        components, cnt = np.unique(labels, return_counts=True)
        if len(components) > 1: # and not args.canonical:
            face_mask = (labels == components[np.argmax(cnt)])
            valid_faces = np.array(a_pose_trimesh.faces)[face_mask, ...]
            n_vertices = len(a_pose_trimesh.vertices)
            vertex_mask = np.isin(np.arange(n_vertices), valid_faces)
            a_pose_trimesh.update_faces(face_mask)
            a_pose_trimesh.update_vertices(vertex_mask)
            # Re-export the processed mesh
            logger.info('Found mesh with floating blobs {}'.format(cano_filename + '.ply'))
            logger.info('Original mesh had {} vertices, reduced to {} vertices after filtering'.format(n_vertices, len(a_pose_trimesh.vertices)))
            a_pose_trimesh.export(cano_filename + '.ply')

        # Run forward skinning network on the extracted mesh points
        coord_min = data.get('points_corr.coord_min').to(device)
        coord_max = data.get('points_corr.coord_max').to(device)
        center = data.get('points_corr.center').to(device)

        coord_min = coord_min[0].detach().cpu().numpy()
        coord_max = coord_max[0].detach().cpu().numpy()
        center = center[0].detach().cpu().numpy()

        padding = (coord_max - coord_min) * 0.05
        p_hat_np = (np.array(a_pose_trimesh.vertices) / 2.0 + 0.5) * 1.1 * (coord_max - coord_min) + coord_min - padding +  center
        a_pose_trimesh.vertices = p_hat_np
        a_pose_trimesh.export(cano_filename + '.ply')

        p_hat_org = torch.from_numpy(p_hat_np).float().to(device).unsqueeze(0)

        with torch.no_grad():
            coord_max = p_hat_org.max(dim=1, keepdim=True)[0]
            coord_min = p_hat_org.min(dim=1, keepdim=True)[0]

            total_size = (coord_max - coord_min).max(dim=-1, keepdim=True)[0]
            scale = torch.clamp(total_size, min=1.6)
            loc = (coord_max + coord_min) / 2

            sc_factor = 1.0 / scale * 1.5

            p_hat_norm = (p_hat_org - loc) * sc_factor
            inp_norm = p_hat_norm

            c = model.encode_inputs(inp_norm, forward=True, scale=scale)

            c_p = model.get_point_features(p_hat_norm, c=c)
            pts_W_fwd = model.decode_w(p_hat_norm, c=c_p, forward=True)
            pts_W_fwd = F.softmax(pts_W_fwd, dim=1).transpose(1, 2)

        skinning_weights_net = pts_W_fwd[0].detach().cpu().numpy()

        # Apply forward LBS to generated posed shape
        trans = data.get('points_corr.trans').cuda()
        root_orient = data.get('points_corr.root_orient').cuda()
        pose_hand = data.get('points_corr.pose_hand').cuda()
        pose_body = data.get('points_corr.pose_body').cuda()
        body = bm(root_orient=root_orient, pose_body=pose_body, pose_hand=pose_hand, trans=trans)
        bone_transforms = body.bone_transforms[0].detach().cpu().numpy()
        Jtr = body.Jtr[0].detach().cpu().numpy()
        Jtr_a_pose = body.Jtr_a_pose[0].detach().cpu().numpy()
        trans = trans[0].detach().cpu().numpy()

        # We set rigid transforms of the hands and feet to be the same as their parents
        # as they are often not accurately registered
        bone_transforms[zero_indices, ...] = bone_transforms[zero_indices_parents, ...]

        T = np.dot(skinning_weights_net, bone_transforms.reshape([-1, 16])).reshape([-1, 4, 4])

        # Compute T such that it transforms points in Vitruvian A-pose to transformed space
        bone_transforms_02v = get_transforms_02v(Jtr_a_pose)
        T_v = np.dot(skinning_weights_net, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        T = np.matmul(T, np.linalg.inv(T_v))

        # Transform mesh points
        n_pts = p_hat_np.shape[0]
        homogen_coord = np.ones([n_pts, 1], dtype=np.float32)
        a_pose_homo = np.concatenate([p_hat_np, homogen_coord], axis=-1).reshape([n_pts, 4, 1])
        body_mesh = np.matmul(T, a_pose_homo)[:, :3, 0].astype(np.float32) + trans

        # Create and save transformed mesh
        posed_trimesh = trimesh.Trimesh(vertices=body_mesh, faces=a_pose_trimesh.faces, process=False)
        posed_trimesh.visual = a_pose_trimesh.visual
        posed_trimesh.export(posed_filename + '.ply')
        # np.save(os.path.join(cloth_dir, filebase + '.pelvis.npy'), Jtr[0])
        logger.info("Exported mesh: {}".format(posed_filename + '.ply'))

    exit(0)
