'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''
#!/usr/bin/env python3

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import torch.nn.functional as F

import depth2mesh.utils.diff_operators as diff_operators

logger = logging.getLogger(__name__)

def create_mesh(
    decoder, thetas=None, filename=None, N=256, max_batch=64 ** 3, offset=None, scale=None, **kwargs
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        # print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        if thetas is not None:
            if isinstance(thetas, dict):
                model_input = thetas
                model_input.update({'coords': sample_subset.unsqueeze(0)})
            else:
                if len(thetas.shape) > 1:
                    full_model = kwargs.get('full_model')
                    skinning_features = kwargs.get('skinning_features')

                    loc = kwargs.get('loc')
                    sc_factor = kwargs.get('sc_factor')

                    coord_max = kwargs.get('coord_max')
                    coord_min = kwargs.get('coord_min')
                    center = kwargs.get('center')

                    if full_model is not None and skinning_features is not None:
                        padding = (coord_max - coord_min) * 0.05
                        sample_subset_norm = sample_subset.unsqueeze(0)
                        sample_subset_norm = (sample_subset_norm / 2.0 + 0.5) * 1.1 * (coord_max - coord_min) + coord_min - padding + center
                        sample_subset_norm = (sample_subset_norm - loc) * sc_factor

                        c_p = full_model.get_point_features(sample_subset_norm, c=skinning_features, forward=True)
                        pts_W_fwd = full_model.decode_w(sample_subset_norm, c=c_p, forward=True)
                        pts_W_fwd = F.softmax(pts_W_fwd, dim=1).transpose(1, 2)[:, :, :23].unsqueeze(-1)
                        pts_pose_attn = torch.matmul(full_model.SMPL_W_dense, pts_W_fwd).squeeze(-1)

                        thetas = thetas.view(1, 23, 4).unsqueeze(1)
                        thetas_in = (pts_pose_attn.unsqueeze(-1) * thetas).reshape(1, sample_subset_norm.size(1), -1).squeeze(0)
                    else:
                        thetas_in = thetas.repeat(sample_subset.size(0), 1)
                else:
                    thetas_in = thetas

                model_input = {'coords': sample_subset.unsqueeze(0), 'times': torch.ones(1, dtype=torch.float32).cuda(), 'cond': thetas_in.unsqueeze(0)}
        else:
            model_input = {'coords': sample_subset.unsqueeze(0), 'times': torch.ones(1, dtype=torch.float32).cuda()}

        samples[head : min(head + max_batch, num_samples), 3] = (
            decoder(model_input)['model_out']
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logger.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logger.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
