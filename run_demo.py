# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse


def load_initial_pose(value: str) -> np.ndarray:
  """Accept a path to a .txt/.npy file or 16 whitespace/comma-separated floats."""
  if os.path.isfile(value):
    if value.endswith('.npy'):
      mat = np.load(value)
    else:
      mat = np.loadtxt(value)
  else:
    vals = [float(x) for x in value.replace(',', ' ').split()]
    if len(vals) != 16:
      raise ValueError(f'--initial_pose expects 16 values (got {len(vals)})')
    mat = np.array(vals, dtype=np.float64)
  return mat.reshape(4, 4)


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--initial_pose', type=str, default=None,
                      help='Skip pose optimisation for frame 0 and use this pose instead. '
                           'Accepts a path to a 4x4 .txt/.npy file, or 16 whitespace/comma-separated floats. '
                           'The matrix must be ob_in_cam where the object origin matches the .obj/.glb file origin.')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      if args.initial_pose is not None:
        logging.info("Using provided initial pose; skipping pose optimisation.")
        init_pose = load_initial_pose(args.initial_pose)
        # Input is ob_in_cam wrt the original mesh origin (as in the .obj/.glb file).
        # Internally pose_last is wrt the centered mesh (vertices shifted by -model_center),
        # so: pose_last = init_pose @ translate(+model_center).
        tf_center_inv = np.eye(4)
        tf_center_inv[:3, 3] = est.model_center
        est.pose_last = torch.as_tensor(
          init_pose @ tf_center_inv, device='cuda', dtype=torch.float32
        )
        pose = init_pose
      else:
        mask = reader.get_mask(0).astype(bool)
        pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

    if debug>=2:
      pass

