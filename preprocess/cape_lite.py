import os
import glob
import argparse
import torch
import trimesh

import pickle as pkl
import numpy as np
import torch.nn.functional as F

from scipy.spatial import cKDTree as KDTree

from multiprocessing import Pool
from functools import partial
from human_body_prior.body_model.body_model import BodyModel
from utils import export_points
from im2mesh.utils.logs import create_logger

SMPL2IPNET_IDX = np.array([11, 12, 13, 11, 3, 8, 11, 1, 6, 11, 1, 6, 0, 11, 11, 0, 5, 10, 4, 9, 2, 7, 2, 7])
IPNET2SMPL_IDX = np.array([12, 7, 20, 4, 18, 16, 8, 21, 5, 19, 17, 0, 1, 2])

parser = argparse.ArgumentParser('Read and create meshes CAPE dataset.')
parser.add_argument('--dataset_path', type=str,
                    help='Path to CAPE dataset.')
parser.add_argument('--subjects', default='00032,00096,00122,00127,00134,00145,00159,00215,02474,03223,03284,03331,03375,03383,03394', type=str, metavar='LIST',
                    help='Subjects of CAPE to use, separated by comma.')
parser.add_argument('--sampling_rate', type=int, default=1,
                    help='Sample every K frame(s).')
parser.add_argument('--bm_path', type=str,
                    help='Path to body model')

parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')

parser.add_argument('--generate_voxels', action='store_true',
                    help='Whether generate voxels input or not')
parser.add_argument('--voxels_resolution', type=int, default=128,
                    help='Resolution of the voxels')

parser.add_argument('--points_folder', type=str,
                    help='Output path for points.')

parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
parser.add_argument('--float16', action='store_true',
                    help='Whether to use half precision.')

"Copied from IFNet/IPNet"
def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list

def voxelize(pc, res, bounds=(-1., 1.), save_path=None):
    grid_points = create_grid_points_from_bounds(bounds[0], bounds[1], res)
    occupancies = np.zeros(len(grid_points), dtype=np.int8)
    kdtree = KDTree(grid_points)
    _, idx = kdtree.query(pc)
    occupancies[idx] = 1

    compressed_occupancies = np.packbits(occupancies)

    return compressed_occupancies
"End of copying"

def process_single_file(vertices, vertices_a_pose, Jtr, root_orient, pose_body, pose_hand, bone_transforms, abs_bone_transforms, trans, frame_name, gender, faces, args):
    body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # Get extents of model.
    bb_min = np.min(vertices, axis=0)
    bb_max = np.max(vertices, axis=0)
    # total_size = np.sqrt(np.square(bb_max - bb_min).sum())
    total_size = max(1.6, (bb_max - bb_min).max())  # just in order to be consistent with IPNet

    # Set the center (although this should usually be the origin already).
    loc = np.array(
        [(bb_min[0] + bb_max[0]) / 2,
         (bb_min[1] + bb_max[1]) / 2,
         (bb_min[2] + bb_max[2]) / 2]
    )
    # Scales all dimensions equally.
    scale = total_size / (1 - args.bbox_padding)

    if args.generate_voxels:
        pc = body_mesh.sample(5000)
        pc = (pc - loc) / scale * 1.5   # just in order to be consistent with IPNet
        # Add some noise to mimic actual scans
        noise = 0.001 * np.random.randn(*pc.shape)
        pc = (pc + noise).astype(np.float32)
        voxels_occ = voxelize(pc, args.voxels_resolution)

    if args.generate_voxels:
        export_points(vertices_a_pose, frame_name, loc, scale, args, bone_transforms=bone_transforms, trans=trans, root_orient=root_orient, pose_body=pose_body, pose_hand=pose_hand, gender=gender, Jtr=Jtr, voxels_occ=voxels_occ)
    else:
        export_points(vertices_a_pose, frame_name, loc, scale, args, bone_transforms=bone_transforms, trans=trans, root_orient=root_orient, pose_body=pose_body, pose_hand=pose_hand, gender=gender, Jtr=Jtr)


def cape_extract(args):
    cape_subjects = args.subjects.split(',')

    if not os.path.exists(args.points_folder):
        os.makedirs(args.points_folder)

    logger, _ = create_logger(args.points_folder)

    faces = np.load(os.path.join(args.dataset_path, 'cape_release/misc/smpl_tris.npy'))
    with open(os.path.join(args.dataset_path, 'cape_release/misc/subj_genders.pkl'), 'rb') as f:
        genders = pkl.load(f)

    for subject in cape_subjects:
        gender = genders[subject]
        subject_dir = os.path.join(args.dataset_path, subject)

        minimal_shape_path = os.path.join(args.dataset_path, 'cape_release', 'minimal_body_shape', subject, subject + '_minimal.npy')
        minimal_shape = np.load(minimal_shape_path)

        bm_path = os.path.join(args.bm_path, gender, 'model.pkl')
        # A-pose joint locations are determined by minimal body shape only
        bm = BodyModel(bm_path=bm_path, num_betas=10, batch_size=1, v_template=minimal_shape).cuda()

        sequences = sorted(glob.glob(os.path.join(subject_dir, '*')))
        sequences = [os.path.basename(sequence) for sequence in sequences]

        # J_regressor = bm.J_regressor.detach().cpu().numpy()
        # Jtr_cano = np.dot(J_regressor, minimal_shape)
        # Jtr_cano = Jtr_cano[IPNET2SMPL_IDX, :]

        for sequence in sequences:
            sequence_dir = os.path.join(subject_dir, sequence)
            frames = sorted(glob.glob(os.path.join(sequence_dir, '*.npz')))
            frames = [os.path.basename(frame) for frame in frames]

            if not os.path.exists(os.path.join(args.points_folder, subject, sequence)):
                os.makedirs(os.path.join(args.points_folder, subject, sequence))

            for f_idx in range(0, len(frames), args.sampling_rate):
                frame = frames[f_idx]

                frame_path = os.path.join(sequence_dir, frame)
                frame_name = frame[:-4]
                # frame_name = os.path.join(subject, sequence, sequence + '.{:06d}'.format(f_idx+1))
                frame_name = os.path.join(subject, sequence, frame_name)

                filename = os.path.join(args.points_folder, frame_name + '.npz')

                if not args.overwrite and os.path.exists(filename):
                    print('Points already exist: %s' % filename)
                    continue

                try:
                    data = np.load(frame_path)
                except Exception:
                    logger.warning('Something wrong with {}'.format(frame_path))
                    continue

                pose_body = torch.Tensor(data['pose'][3:66]).view(1, -1).cuda()
                pose_hand = torch.Tensor(data['pose'][66:72]).view(1, -1).cuda()
                root_orient = torch.Tensor(data['pose'][:3]).view(1, -1).cuda()
                trans = torch.Tensor(data['transl']).view(1, -1).cuda()
                v_cano = torch.Tensor(data['v_cano']).view(1, 6890, 3).cuda()

                with torch.no_grad():
                    body = bm(root_orient=root_orient, pose_body=pose_body, pose_hand=pose_hand, trans=trans, clothed_v_template=v_cano)

                    bone_transforms = body.bone_transforms.detach().cpu().numpy()
                    abs_bone_transforms = body.abs_bone_transforms.detach().cpu().numpy()

                    pose_body = pose_body.detach().cpu().numpy()
                    pose_hand = pose_hand.detach().cpu().numpy()
                    Jtr = body.Jtr.detach().cpu().numpy()
                    v_cano = body.v_a_pose.detach().cpu().numpy()
                    v_posed = body.v.detach().cpu().numpy()
                    trans = trans.detach().cpu().numpy()
                    root_orient = root_orient.detach().cpu().numpy()

                process_single_file(v_posed[0], v_cano[0], Jtr[0], root_orient[0], pose_body[0], pose_hand[0], bone_transforms[0], abs_bone_transforms[0], trans[0], frame_name, gender, faces, args)

        del bm


def main(args):
    cape_extract(args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
