import os

import numpy as np

def export_points(a_pose_mesh_points, framename, loc, scale, args, **kwargs):
    kwargs_new = {}
    for k, v in kwargs.items():
        if v is not None:
            kwargs_new[k] = v

    filename = os.path.join(args.points_folder, framename + '.npz')

    if not args.overwrite and os.path.exists(filename):
        print('Points already exist: %s' % filename)
        return

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    # mesh_points = mesh_points.astype(dtype)
    a_pose_mesh_points = a_pose_mesh_points.astype(dtype)

    print('Writing points: %s' % filename)
    np.savez(filename,
             a_pose_mesh_points=a_pose_mesh_points,
             loc=loc, scale=scale,
             **kwargs_new)
