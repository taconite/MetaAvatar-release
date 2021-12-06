import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import glob
import numpy as np
import trimesh
import igl

import pickle as pkl
from torch.utils import data

from scipy.spatial.transform import Rotation as R

# Parts to exclude for CAPE meshes: hands and feet
exclude_indices = np.array([10, 11, 22, 23], dtype=np.int64)
exclude_indices_parents = np.array([7, 8, 20, 21], dtype=np.int64)

# Parts to exclude for raw-scans: hands, wrists, feet and ankles
exclude_more_indices = np.array([7, 8, 10, 11, 20, 21, 22, 23], dtype=np.int64)
exclude_more_indices_parents = np.array([4, 5, 4, 5, 18, 19, 18, 19], dtype=np.int64)

''' Copied from IPNet'''
def get_3DSV(mesh):
    from opendr.camera import ProjectPoints
    from opendr.renderer import DepthRenderer
    WIDTH, HEIGHT = 250, 250

    rt = R.from_euler('xyz', [np.pi, 0, 0]).as_rotvec()
    camera = ProjectPoints(v=mesh.vertices,
                           f=np.array([WIDTH, WIDTH]),
                           c=np.array([WIDTH, HEIGHT]) / 2.,
                           t=np.array([0, 0, 3.0]),
                           rt=rt,
                           k=np.zeros(5))
    frustum = {'near': 1., 'far': 10., 'width': WIDTH, 'height': HEIGHT}
    rn = DepthRenderer(camera=camera, frustum=frustum, f=mesh.faces, overdraw=False)

    # import cv2
    depth_image = rn.depth_image.copy()
    mask = depth_image < depth_image.max() - 0.01
    depth_image[~mask] = 0
    depth_image[mask] = 255 - (depth_image[mask] - depth_image[mask].min()) / (depth_image[mask].max() - depth_image[mask].min()) * 255

    points3d = camera.unproject_depth_image(rn.r)
    mask = points3d[:, :, 2] > np.min(points3d[:, :, 2]) + 0.01 # mask for foreground pixels

    points3d = points3d[mask]

    return points3d, depth_image
''' End of copying'''

class CAPECorrDataset(data.Dataset):
    ''' CAPE dataset class.
    '''

    def __init__(self, dataset_folder,
                 raw_scan_folder='/home/sfwang/Datasets/CAPE_raw_scans',
                 subjects=['00032', '00096', '00159', '03223'],
                 mode='train',
                 use_aug=False,
                 input_pointcloud_n=5000,
                 input_pointcloud_noise=0.001,
                 use_raw_scans=True,
                 normalized_scale=False,
                 action_names=None,
                 cloth_types=None,
                 subject_idx=None,
                 subsampling_rate=1,
                 start_offset=0,
                 keep_aspect_ratio=True,
                 single_view=True):
        ''' Initialization of the CAPECorrDataset instance.

        Args:
            dataset_folder (str): folder that stores processed, registered models
            raw_scan_folder (str): folder that sotres raw scans
            subjects (list of strs): which subjects to include in this dataset instance
            mode (str): mode of the dataset. Can be either 'train', 'val' or 'test'
            use_aug (bool): whether to use data augmentation or not. Only relevent to 'train' mode
            input_pointcloud_n (int): number for points to sample from each scan
            input_pointcloud_noise (float): noise level for sampled points
            use_raw_scans (bool): whether to use raw scan or not. If True, will read raw scans from
                                  raw_scan_folder
            normalized_scale (bool): normalize all points into [-1, 1]
            action_names (list of str): which action(s) to load. If None, will load all actions
            cloth_types (list of str): which cloth type(s) to load. If None, will load all cloth types
            subject_idx (int): use only the speficied subject. If None, use all subjects
            subsampling_rate (int): subsampling rate for subsampling dataset frames
            start_offset (int): first index for sampling the dataset
            keep_aspect_ratio (bool): whether to keep aspect ratio when normalizing canonical space
            single_view (bool): whether to generate single-view point clouds or full-body point clouds
        '''
        # Attributes
        self.cape_path = '/home/sfwang/Datasets/CAPE'
        self.dataset_folder = dataset_folder
        self.raw_scan_folder = raw_scan_folder
        self.use_aug = use_aug
        self.mode = mode
        self.normalized_scale = normalized_scale
        self.use_raw_scans = use_raw_scans

        self.input_pointcloud_n = input_pointcloud_n
        self.input_pointcloud_noise = input_pointcloud_noise

        self.faces = np.load('body_models/misc/faces.npz')['faces']
        self.skinning_weights = dict(np.load('body_models/misc/skinning_weights_all.npz'))
        self.posedirs = dict(np.load('body_models/misc/posedirs_all.npz'))
        self.J_regressor = dict(np.load('body_models/misc/J_regressors.npz'))

        self.v_templates = dict(np.load('body_models/misc/v_templates.npz'))

        self.keep_aspect_ratio = keep_aspect_ratio
        self.single_view = single_view

        with open(os.path.join(self.cape_path, 'cape_release/misc/subj_genders.pkl'), 'rb') as f:
            genders = pkl.load(f)

        self.rot45p = R.from_euler('z', 45, degrees=True).as_matrix()
        self.rot45n = R.from_euler('z', -45, degrees=True).as_matrix()

        # Get all data
        self.data = []
        self.indices = []
        if subject_idx is not None:
            subjects = [subjects[idx] for idx in subject_idx] if isinstance(subject_idx, list) else [subjects[subject_idx]]

        roll= 0
        pitch = 0
        yaw = 0

        for subject in subjects:
            subject_dir = os.path.join(dataset_folder, subject)
            subset = 'cape'
            if cloth_types is None:
                sequence_dirs = glob.glob(os.path.join(subject_dir, '*'))
                sequences = set()
                for sequence_dir in sequence_dirs:
                    sequences.add(os.path.basename(sequence_dir).split('.')[0])
            else:
                sequences = set()
                for cloth_type in cloth_types:
                    if action_names is None:
                        sequence_dirs = glob.glob(os.path.join(subject_dir, cloth_type + '_*'))
                        for sequence_dir in sequence_dirs:
                            sequences.add(os.path.basename(sequence_dir).split('.')[0])
                    else:
                        for action_name in action_names:
                            sequence_dir = os.path.join(subject_dir, cloth_type + '_' + action_name)
                            assert (os.path.exists(sequence_dir))
                            sequences.add(os.path.basename(sequence_dir).split('.')[0])

            sequences = sorted(list(sequences))

            group_indices = []
            curr_cloth_type = None
            for sequence in sequences:
                if use_raw_scans:
                    assert (os.path.exists(os.path.join(self.raw_scan_folder, subject, sequence)))

                models_dir = os.path.join(subject_dir, sequence)
                model_files = sorted(glob.glob(os.path.join(models_dir, '*.npz')))

                if subsampling_rate > len(model_files) // 2:
                    # If we can only take at most 2 frame per sequence
                    model_files = model_files[len(model_files) // 2::subsampling_rate]
                else:
                    # model_files = model_files[subsampling_rate::subsampling_rate]
                    if start_offset <= 0:
                        model_files = model_files[::subsampling_rate]
                    else:
                        model_files = model_files[start_offset::subsampling_rate]

                if use_raw_scans:
                    scan_files = [os.path.basename(model_file)[:-4] + '.ply' for model_file in model_files]
                    scan_files = [os.path.join(self.raw_scan_folder, subject, sequence, scan_file) for scan_file in scan_files]
                else:
                    scan_files = model_files

                cloth_type = sequence.split('_')[0]
                if curr_cloth_type is None:
                    curr_cloth_type = cloth_type

                if cloth_type != curr_cloth_type:
                    assert(len(group_indices) > 0)

                    curr_cloth_type = cloth_type

                    self.indices.append(group_indices)
                    group_indices = []

                for idx, (scan_file, model_file) in enumerate(zip(scan_files, model_files)):
                    assert (os.path.exists(scan_file))

                    group_indices.append(len(self.data))

                    if self.mode in ['val', 'test'] and self.single_view:
                        pitch = (pitch + 45) % 360  # Increate pitch angle by 45 for every frame at test-time
                        self.data.append(
                                {'subset': subset,
                                 'subject': subject,
                                 'gender': genders[subject],
                                 'sequence': sequence,
                                 'cloth_type': cloth_type,
                                 'scan_path': scan_file if use_raw_scans else None,
                                 'data_path': model_file,
                                 'frame_idx': idx,
                                 'roll': roll,
                                 'pitch': pitch,
                                 'yaw': yaw}
                                )
                    else:
                        self.data.append(
                                {'subset': subset,
                                 'subject': subject,
                                 'gender': genders[subject],
                                 'sequence': sequence,
                                 'frame_idx': idx,
                                 'cloth_type': cloth_type,
                                 'scan_path': scan_file if use_raw_scans else None,
                                 'data_path': model_file}
                                )

            # assert(len(group_indices) > 0)
            self.indices.append(group_indices)

    def map_mesh_points_to_reference(self, pts, src_verts, src_faces, ref_cano, ref_hat, ref_W):
        """ Finds closest points to pts on mesh (represented by src_verts and src_faces), then maps
            the closest points on mesh to reference space.

        Args:
            pts (N x 3 float numpy array): query points in transformed space
            src_verts (6890 x 3 float numpy array): SMPL vertices in transformed space
            src_faces (6890 x 3 float numpy array): SMPL faces
            ref_cano (6890 x 3 float numpy array): template SMPL vertices in canonical space
            ref_hat (6890 x 3 float numpy array): SMPL vertices in canonical space, with shape and
                                                  pose correctives
            ref_W (6890 x 24 float numpy array): skinning weights for each SMPL vertex

        Returns:
            corr_cano (N x 3 float numpy array): correspondences of pts in template canonical space
            corr_hat (N x 3 float numpy array): correspondences of pts in canonical space, with shape
                                                and pose correctives
            pts_W (N x 24 float numpy array): skinning weights of pts
            mask (N float numpy array): mask indicating which points in pts are close enough to mesh
            closest_faces (N x 3 int numpy array): closest triangle faces in src_faces to pts
            closest_points (N x 3 float numpy array): closest points on mesh to pts
        """
        closest_dists, closest_faces, closest_points = igl.point_mesh_squared_distance(pts, src_verts, src_faces)
        bary_coords = igl.barycentric_coordinates_tri(
                                                  closest_points,
                                                  src_verts[src_faces[closest_faces, 0], :],
                                                  src_verts[src_faces[closest_faces, 1], :],
                                                  src_verts[src_faces[closest_faces, 2], :]
                                                 )
        vert_ids = src_faces[closest_faces, ...]
        corr_cano = (ref_cano[vert_ids] * bary_coords[..., np.newaxis]).sum(axis=1)
        corr_hat = (ref_hat[vert_ids] * bary_coords[..., np.newaxis]).sum(axis=1)
        pts_W = (ref_W[vert_ids] * bary_coords[..., np.newaxis]).sum(axis=1)

        # Mask out outliers that belongs to ground, walls, etc.
        # Note that in DFaust, 93% of scan points are within 1mm of the registered mesh
        # here we use 2cm for thresholding, which should be more than enough
        mask = np.linalg.norm(closest_dists[..., np.newaxis] - pts, axis=-1) < 0.02

        return corr_cano, corr_hat, pts_W, mask, closest_faces, closest_points

    def augm_params(self, roll_range=10, pitch_range=180, yaw_range=10):
        """ Get augmentation parameters.

        Args:
            roll_range (int): roll angle sampling range (train mode) or value (test mode)
            pitch_range (int): pitch angle sampling range (train mode) or value (test mode)
            yaw_range (int): yaw angle sampling range (train mode) or value (test mode)

        Returns:
            rot_mat (4 x 4 float numpy array): homogeneous rotation augmentation matrix.
        """
        if self.mode == 'train' and self.use_aug:
            # Augmentation during training

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            # Roll
            rot_x = min(2*roll_range,
                    max(-2*roll_range, np.random.randn()*roll_range))

            sn, cs = np.sin(np.pi / 180 * rot_x), np.cos(np.pi / 180 * rot_x)
            rot_x = np.eye(4)
            rot_x[1, 1] = cs
            rot_x[1, 2] = -sn
            rot_x[2, 1] = sn
            rot_x[2, 2] = cs
            # but it is identity with probability 3/5
            if np.random.uniform() <= 0.6:
                rot_x = np.eye(4)

            rot_y = min(2*pitch_range,
                    max(-2*pitch_range, (np.random.rand() * 2 - 1)*pitch_range))

            # Pitch
            sn, cs = np.sin(np.pi / 180 * rot_y), np.cos(np.pi / 180 * rot_y)
            rot_y = np.eye(4)
            rot_y[0, 0] = cs
            rot_y[0, 2] = sn
            rot_y[2, 0] = -sn
            rot_y[2, 2] = cs

            rot_z = min(2*yaw_range,
                    max(-2*yaw_range, np.random.randn()*yaw_range))

            # Yaw
            sn, cs = np.sin(np.pi / 180 * rot_z), np.cos(np.pi / 180 * rot_z)
            rot_z = np.eye(4)
            rot_z[0, 0] = cs
            rot_z[0, 1] = -sn
            rot_z[1, 0] = sn
            rot_z[1, 1] = cs
            # but it is identity with probability 3/5
            if np.random.uniform() <= 0.6:
                rot_z = np.eye(4)

            rot_mat = np.dot(rot_x, np.dot(rot_y, rot_z))
        elif self.mode in ['test'] and self.single_view:
            # Simulate a rotating camera

            # Roll
            rot_x = roll_range

            sn, cs = np.sin(np.pi / 180 * rot_x), np.cos(np.pi / 180 * rot_x)
            rot_x = np.eye(4)
            rot_x[1, 1] = cs
            rot_x[1, 2] = -sn
            rot_x[2, 1] = sn
            rot_x[2, 2] = cs

            rot_y = pitch_range

            # Pitch
            sn, cs = np.sin(np.pi / 180 * rot_y), np.cos(np.pi / 180 * rot_y)
            rot_y = np.eye(4)
            rot_y[0, 0] = cs
            rot_y[0, 2] = sn
            rot_y[2, 0] = -sn
            rot_y[2, 2] = cs

            rot_z = yaw_range

            # Yaw
            sn, cs = np.sin(np.pi / 180 * rot_z), np.cos(np.pi / 180 * rot_z)
            rot_z = np.eye(4)
            rot_z[0, 0] = cs
            rot_z[0, 1] = -sn
            rot_z[1, 0] = sn
            rot_z[1, 1] = cs

            rot_mat = np.dot(rot_x, np.dot(rot_y, rot_z))
        else:
            # No augmentation
            rot_mat = np.eye(4)


        return rot_mat

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        data_path = self.data[idx]['data_path']
        scan_path = self.data[idx]['scan_path']
        subject = self.data[idx]['subject']
        gender = self.data[idx]['gender']

        data = {}

        if not self.single_view or self.mode == 'train':
            # For training mode, generate random rotation augmentation
            aug_rot = self.augm_params().astype(np.float32)
        else:
            # For testing in single-view mode, load predefined set of rotations
            roll = self.data[idx]['roll']
            pitch = self.data[idx]['pitch']
            yaw = self.data[idx]['yaw']
            aug_rot = self.augm_params(roll, pitch, yaw).astype(np.float32)

        points_dict = np.load(data_path)

        # Load registered models and (optionally) raw scans
        if self.use_raw_scans:
            raw_trimesh = trimesh.load(scan_path)
            if np.max(raw_trimesh.vertices) > 10:
                raw_trimesh.vertices /= 1000 # mm to m

        body_mesh_a_pose = points_dict['a_pose_mesh_points']
        # Break symmetry if given in float16:
        if body_mesh_a_pose.dtype == np.float16:
            body_mesh_a_pose = body_mesh_a_pose.astype(np.float32)
            body_mesh_a_pose += 1e-4 * np.random.randn(*body_mesh_a_pose.shape)
        else:
            body_mesh_a_pose = body_mesh_a_pose.astype(np.float32)

        a_pose_trimesh = trimesh.Trimesh(vertices=body_mesh_a_pose, faces=self.faces, process=False)

        n_smpl_points = body_mesh_a_pose.shape[0]
        bone_transforms_org = points_dict['bone_transforms'].astype(np.float32)
        bone_transforms = bone_transforms_org.copy()
        trans = points_dict['trans'].astype(np.float32)

        body_mesh_a_pose_0 = body_mesh_a_pose - trans   # body_mesh_a_pose_0 includes global translation, so need to subtract it

        # Also get GT SMPL poses
        root_orient = points_dict['root_orient'].astype(np.float32)
        pose_body = points_dict['pose_body'].astype(np.float32)
        pose_hand = points_dict['pose_hand'].astype(np.float32)
        pose = np.concatenate([pose_body, pose_hand], axis=-1)

        pose = R.from_rotvec(pose.reshape([-1, 3]))
        pose_quat = pose.as_quat()

        pose_quat = pose_quat.reshape(-1)
        pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose.as_matrix()], axis=0).reshape([-1, 9])   # 24 x 9

        # Minimally clothed shape
        minimal_shape_path = os.path.join(self.cape_path, 'cape_release', 'minimal_body_shape', subject, subject + '_minimal.npy')

        posedir = self.posedirs[gender]
        J_regressor = self.J_regressor[gender]
        minimal_shape = np.load(minimal_shape_path)
        Jtr = np.dot(J_regressor, minimal_shape)

        pose_mat = pose.as_matrix()
        ident = np.eye(3)
        pose_feature = (pose_mat - ident).reshape([207, 1])
        pose_offsets = np.dot(posedir.reshape([-1, 207]), pose_feature).reshape([6890, 3])
        minimal_shape += pose_offsets

        # Get posed clothed and minimally-clothed SMPL meshes
        skinning_weights = self.skinning_weights[gender]
        T = np.dot(skinning_weights, bone_transforms.reshape([-1, 16])).reshape([-1, 4, 4])

        homogen_coord = np.ones([n_smpl_points, 1], dtype=np.float32)
        a_pose_homo = np.concatenate([body_mesh_a_pose_0, homogen_coord], axis=-1).reshape([n_smpl_points, 4, 1])
        body_mesh = (np.matmul(T, a_pose_homo)[:, :3, 0].astype(np.float32) + trans).astype(np.float32)

        a_pose_homo = np.concatenate([minimal_shape, homogen_coord], axis=-1).reshape([n_smpl_points, 4, 1])
        minimal_body_mesh = np.matmul(T, a_pose_homo)[:, :3, 0].astype(np.float32) + trans
        if self.mode in ['val', 'test']:
            minimal_body_vertices = minimal_body_mesh.copy()

        # Apply rotation augmentation
        center = body_mesh.mean(0)
        if self.use_raw_scans:
            scan_vertices = np.array(raw_trimesh.vertices)
            scan_vertices = np.dot(aug_rot[:3, :3], (scan_vertices - center).T).T + center
            raw_trimesh = trimesh.Trimesh(vertices=scan_vertices, faces=raw_trimesh.faces, process=False)
            body_mesh = np.dot(aug_rot[:3, :3], (body_mesh - center).T).T + center
            minimal_body_mesh = np.dot(aug_rot[:3, :3], (minimal_body_mesh - center).T).T + center
        else:
            body_mesh = np.dot(aug_rot[:3, :3], (body_mesh - center).T).T + center
            minimal_body_mesh = np.dot(aug_rot[:3, :3], (minimal_body_mesh - center).T).T + center

        bone_transforms[:, :3, -1] += trans - center
        bone_transforms = np.matmul(np.expand_dims(aug_rot, axis=0), bone_transforms)
        bone_transforms[:, :3, -1] += center

        posed_trimesh = trimesh.Trimesh(vertices=body_mesh, faces=self.faces)
        minimal_posed_trimesh = trimesh.Trimesh(vertices=minimal_body_mesh, faces=self.faces)

        # Sample points from mesh
        if self.single_view:
            if self.use_raw_scans:
                points_corr, depth_image = get_3DSV(raw_trimesh)
                points_corr_reg, _, _ = get_3DSV(posed_trimesh)
            else:
                points_corr, depth_image = get_3DSV(posed_trimesh)

            normals = None  # we don't use normals for single-view setup

            if not self.use_raw_scans:
                noise = self.input_pointcloud_noise * np.random.randn(*points_corr.shape)
                points_corr = (points_corr + noise).astype(np.float32)
        else:
            if self.use_raw_scans:
                points_corr, face_idx = raw_trimesh.sample(self.input_pointcloud_n, return_index=True)
                normals = raw_trimesh.face_normals[face_idx.ravel()]

                points_corr_reg, face_idx_reg = posed_trimesh.sample(self.input_pointcloud_n, return_index=True)
                normals_reg = posed_trimesh.face_normals[face_idx_reg.ravel()]
            else:
                points_corr, face_idx = posed_trimesh.sample(self.input_pointcloud_n, return_index=True)
                noise = self.input_pointcloud_noise * np.random.randn(*points_corr.shape)
                points_corr = (points_corr + noise).astype(np.float32)
                normals = posed_trimesh.face_normals[face_idx.ravel()]

        points_corr_org = points_corr.copy()

        # Specify the bone transformations that transform a SMPL A-pose mesh
        # to a star-shaped A-pose (i.e. Vitruvian A-pose)
        bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))

        # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
        chain = [1, 4, 7, 10]
        rot = self.rot45p.copy()
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
        rot = self.rot45n.copy()
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
        bone_transforms_02v_org = bone_transforms_02v.copy()

        T = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        body_mesh_v_pose_0 = np.matmul(T[:, :3, :3], body_mesh_a_pose_0[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]
        minimal_shape_v = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        points_corr_cano, points_corr_hat, _, mask, closest_faces, _ = self.map_mesh_points_to_reference(points_corr, body_mesh, self.faces, self.v_templates[gender], body_mesh_v_pose_0, skinning_weights)
        _, _, pts_W, _, _, _ = self.map_mesh_points_to_reference(points_corr, minimal_body_mesh, self.faces, self.v_templates[gender], minimal_shape, skinning_weights)

        if self.use_raw_scans:
            points_corr_cano_reg, points_corr_hat_reg, pts_W_reg, _, _, _ = self.map_mesh_points_to_reference(points_corr_reg, minimal_body_mesh, self.faces, self.v_templates[gender], minimal_shape, skinning_weights)
            mask = mask & ~np.isin(pts_W.argmax(-1), exclude_more_indices)
            reg_mask = np.isin(pts_W_reg.argmax(-1), exclude_more_indices)
            points_corr_reg = points_corr_reg[reg_mask]
            points_corr_cano_reg = points_corr_cano_reg[reg_mask]
            points_corr_hat_reg = points_corr_hat_reg[reg_mask]
            pts_W_reg = pts_W_reg[reg_mask]
            normals_reg = normals_reg[reg_mask]

        if not self.use_raw_scans:
            mask = np.ones(mask.shape, dtype=np.bool)
            normals_a_pose = a_pose_trimesh.face_normals[closest_faces.ravel()]
        else:
            normals_a_pose = None

        points_corr = points_corr[mask]
        points_corr_cano = points_corr_cano[mask]
        points_corr_hat = points_corr_hat[mask]
        points_corr_org = points_corr_org[mask]
        if normals is not None:
            normals = normals[mask]
        if normals_a_pose is not None:
            normals_a_pose = normals_a_pose[mask]

        pts_W = pts_W[mask]

        if self.use_raw_scans:
            # Replace hands&feet of raw-scans with SMPL hands&feet
            points_corr = np.concatenate([points_corr, points_corr_reg], axis=0)
            points_corr_cano = np.concatenate([points_corr_cano, points_corr_cano_reg], axis=0)
            points_corr_hat = np.concatenate([points_corr_hat, points_corr_hat_reg], axis=0)
            points_corr_org = points_corr.copy()
            pts_W = np.concatenate([pts_W, pts_W_reg], axis=0)
            normals = np.concatenate([normals, normals_reg], axis=0)

        if self.input_pointcloud_n <= points_corr.shape[0]:
            rand_inds = np.random.choice(points_corr.shape[0], size=self.input_pointcloud_n, replace=False)
        else:
            rand_inds = np.random.choice(points_corr.shape[0], size=self.input_pointcloud_n, replace=True)

        points_corr = points_corr[rand_inds, :]
        points_corr_cano = points_corr_cano[rand_inds, :]
        points_corr_hat = points_corr_hat[rand_inds, :]
        if normals is not None:
            normals = normals[rand_inds, :]

        if normals_a_pose is not None:
            normals_a_pose = normals_a_pose[rand_inds, :]

        pts_W = pts_W[rand_inds, :]

        # Get extents of point cloud.
        bb_min = np.min(points_corr, axis=0)
        bb_max = np.max(points_corr, axis=0)
        # total_size = np.sqrt(np.square(bb_max - bb_min).sum())
        total_size = (bb_max - bb_min).max()
        # Scales all dimensions equally.
        scale = max(1.6, total_size)    # 1.6 is the magic number from IPNet
        loc = np.array(
            [(bb_min[0] + bb_max[0]) / 2,
             (bb_min[1] + bb_max[1]) / 2,
             (bb_min[2] + bb_max[2]) / 2],
            dtype=np.float32
        )

        # If normalized_scale is true, the scan points would be in [-0.75, 0.75]^3
        sc_factor = 1.0 / scale * 1.5 if self.normalized_scale else 1.0 # 1.5 is the magic number from IPNet
        offset = loc

        # Normalize conanical pose points with GT full-body scales. This should be fine as
        # at test time we register each frame first, thus obtaining full-body scale
        center = np.mean(minimal_shape_v, axis=0)
        # center = np.zeros(3, dtype=np.float32)
        minimal_shape_v_centered = minimal_shape_v - center
        points_corr_hat -= center
        if self.keep_aspect_ratio:
            coord_max = minimal_shape_v_centered.max()
            coord_min = minimal_shape_v_centered.min()
        else:
            coord_max = minimal_shape_v_centered.max(axis=0, keepdims=True)
            coord_min = minimal_shape_v_centered.min(axis=0, keepdims=True)

        padding = (coord_max - coord_min) * 0.05
        points_corr_hat = (points_corr_hat - coord_min + padding) / (coord_max - coord_min) / 1.1
        points_corr_hat -= 0.5
        points_corr_hat *= 2.

        Jtr_norm = Jtr - center
        Jtr_norm = (Jtr_norm - coord_min + padding) / (coord_max - coord_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.

        bone_transforms[:, :3, -1] -= offset
        bone_transforms[:, :3, -1] *= sc_factor
        bone_transforms_02v[:, :3, -1] *= sc_factor

        rand_inds = np.random.choice(6890, size=5000, replace=False)
        minimal_shape_samples = minimal_shape_v[rand_inds, :]

        data = {
            None: (points_corr.astype(np.float32) - offset) * sc_factor,
            'cano': points_corr_cano.astype(np.float32) * sc_factor,
            'a_pose': points_corr_hat.astype(np.float32),
            'skinning_weights': pts_W.astype(np.float32),
            'loc': loc.astype(np.float32),
            'scale': np.array(scale, dtype=np.float32),
            'trans': trans.astype(np.float32),
            'pose': pose_quat.astype(np.float32),
            'bone_transforms': bone_transforms.astype(np.float32),
            'bone_transforms_org': bone_transforms_org.astype(np.float32),
            'bone_transforms_02v': bone_transforms_02v.astype(np.float32),
            'bone_transforms_02v_org': bone_transforms_02v_org.astype(np.float32),
            'coord_max': coord_max.astype(np.float32),
            'coord_min': coord_min.astype(np.float32),
            'center': center.astype(np.float32),
            'minimal_shape': minimal_shape_v.astype(np.float32),
            'minimal_shape_samples': minimal_shape_samples.astype(np.float32),
            'root_orient': root_orient,
            'pose_hand': pose_hand,
            'pose_body': pose_body,
            'rots': pose_rot.astype(np.float32),
            'Jtrs': Jtr_norm.astype(np.float32),
        }

        if not self.single_view:
            data.update({'normals': normals.astype(np.float32)})

        if not self.use_raw_scans:
            data.update({'normals_a_pose': normals_a_pose.astype(np.float32)})

        if self.mode in ['val', 'test']:
            bone_transforms_ex = bone_transforms_org.copy()
            if self.use_raw_scans:
                bone_transforms_ex[exclude_more_indices, ...] = bone_transforms_ex[exclude_more_indices_parents, ...]
            else:
                bone_transforms_ex[exclude_indices, ...] = bone_transforms_ex[exclude_indices_parents, ...]

            T = np.dot(skinning_weights, bone_transforms_ex.reshape([-1, 16])).reshape([-1, 4, 4])

            homogen_coord = np.ones([n_smpl_points, 1], dtype=np.float32)
            a_pose_homo = np.concatenate([body_mesh_a_pose_0, homogen_coord], axis=-1).reshape([n_smpl_points, 4, 1])
            body_mesh = (np.matmul(T, a_pose_homo)[:, :3, 0].astype(np.float32) + trans).astype(np.float32)

            data.update({'minimal_vertices': minimal_body_vertices,
                         'smpl_vertices': body_mesh,
                         'smpl_vertices_a_pose': body_mesh_a_pose})

        data_out = {}
        field_name = 'points_corr'
        for k, v in data.items():
            if k is None:
                data_out[field_name] = v
            else:
                data_out['%s.%s' % (field_name, k)] = v

        data_out.update(
            {'inputs': (points_corr.astype(np.float32) - offset) * sc_factor,
             'idx': idx,
            }
        )

        if self.mode in ['val', 'test']:
            data_out.update({'gender': gender})

        if self.single_view and self.mode in ['val', 'test']:
            data_out.update({'depth_image': depth_image.astype(np.uint8)})

        return data_out

    def get_model_dict(self, idx):
        return self.data[idx]
