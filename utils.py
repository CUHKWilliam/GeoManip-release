import os
import numpy as np
from numba import njit
import open3d as o3d
import datetime
import scipy.interpolate as interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline
import transform_utils as T
import yaml
import cv2
import base64


# ===============================================
# = optimization utils
# ===============================================
def normalize_vars(vars, og_bounds):
    """
    Given 1D variables and bounds, normalize the variables to [-1, 1] range.
    """
    normalized_vars = np.empty_like(vars)
    for i, (b_min, b_max) in enumerate(og_bounds):
        normalized_vars[i] = (vars[i] - b_min) / (b_max - b_min) * 2 - 1
    return normalized_vars

def unnormalize_vars(normalized_vars, og_bounds):
    """
    Given 1D variables in [-1, 1] and original bounds, denormalize the variables to the original range.
    """
    vars = np.empty_like(normalized_vars)
    for i, (b_min, b_max) in enumerate(og_bounds):
        vars[i] = (normalized_vars[i] + 1) / 2 * (b_max - b_min) + b_min
    return vars

def calculate_collision_cost(poses, sdf_func, collision_points, threshold):
    assert poses.shape[1:] == (4, 4)
    transformed_pcs = batch_transform_points(collision_points, poses)
    transformed_pcs_flatten = transformed_pcs.reshape(-1, 3)  # [num_poses * num_points, 3]
    signed_distance = sdf_func(transformed_pcs_flatten) + threshold  # [num_poses * num_points]
    signed_distance = signed_distance.reshape(-1, collision_points.shape[0])  # [num_poses, num_points]
    non_zero_mask = signed_distance > 0
    collision_cost = np.sum(signed_distance[non_zero_mask])
    return collision_cost

@njit(cache=True, fastmath=True)
def consistency(poses_a, poses_b, rot_weight=0.5):
    assert poses_a.shape[1:] == (4, 4) and poses_b.shape[1:] == (4, 4), 'poses must be of shape (N, 4, 4)'
    min_distances = np.zeros(len(poses_a), dtype=np.float64)
    for i in range(len(poses_a)):
        min_distance = 9999999
        a = poses_a[i]
        for j in range(len(poses_b)):
            b = poses_b[j]
            pos_distance = np.linalg.norm(a[:3, 3] - b[:3, 3])
            rot_distance = angle_between_rotmat(a[:3, :3], b[:3, :3])
            distance = pos_distance + rot_distance * rot_weight
            min_distance = min(min_distance, distance)
        min_distances[i] = min_distance
    return np.mean(min_distances)

def transform_keypoints(transform, keypoints, movable_mask):
    assert transform.shape == (4, 4)
    transformed_keypoints = keypoints.copy()
    if movable_mask.sum() > 0:
        transformed_keypoints[movable_mask] = np.dot(keypoints[movable_mask], transform[:3, :3].T) + transform[:3, 3]
    return transformed_keypoints
import copy
def transform_geometry(transform, part_pts_dict, moving_part_names, pos_only=False):
    import utils
    part_pts_dict_latest = copy.deepcopy(part_pts_dict[-1])
    part_pts_dict2 = copy.deepcopy(utils.ENV.part_to_pts_dict)
    for part_name in part_pts_dict_latest.keys():
        if part_name in moving_part_names:
            part_pts = part_pts_dict_latest[part_name]
            if not pos_only:
                part_pts_dict_latest[part_name] = np.dot(part_pts, transform[:3, :3].T) + transform[:3, 3]
            else:
                part_pts_dict_latest[part_name] = part_pts + transform[:3, 3]
    part_pts_dict2.append(part_pts_dict_latest)
    return part_pts_dict2

@njit(cache=True, fastmath=True)
def batch_transform_points(points, transforms, pos_only=False):
    """
    Apply multiple of transformation to point cloud, return results of individual transformations.
    Args:
        points: point cloud (N, 3).
        transforms: M 4x4 transformations (M, 4, 4).
    Returns:
        np.array: point clouds (M, N, 3).
    """
    assert transforms.shape[1:] == (4, 4), 'transforms must be of shape (M, 4, 4)'
    transformed_points = np.zeros((transforms.shape[0], points.shape[0], 3))
    for i in range(transforms.shape[0]):
        pos, R = transforms[i, :3, 3], transforms[i, :3, :3]
        if not pos_only:
            transformed_points[i] = np.dot(points, R.T) + pos
        else:
            transformed_points[i] = points + pos
    return transformed_points

@njit(cache=True, fastmath=True)
def get_samples_jitted(control_points_homo, control_points_quat, opt_interpolate_pos_step_size, opt_interpolate_rot_step_size):
    assert control_points_homo.shape[1:] == (4, 4)
    # calculate number of samples per segment
    num_samples_per_segment = np.empty(len(control_points_homo) - 1, dtype=np.int64)
    for i in range(len(control_points_homo) - 1):
        start_pos = control_points_homo[i, :3, 3]
        start_rotmat = control_points_homo[i, :3, :3]
        end_pos = control_points_homo[i+1, :3, 3]
        end_rotmat = control_points_homo[i+1, :3, :3]
        pos_diff = np.linalg.norm(start_pos - end_pos)
        rot_diff = angle_between_rotmat(start_rotmat, end_rotmat)
        pos_num_steps = np.ceil(pos_diff / opt_interpolate_pos_step_size)
        rot_num_steps = np.ceil(rot_diff / opt_interpolate_rot_step_size)
        num_path_poses = int(max(pos_num_steps, rot_num_steps))
        num_path_poses = max(num_path_poses, 2)  # at least 2 poses, start and end
        num_samples_per_segment[i] = num_path_poses
    # fill in samples
    num_samples = num_samples_per_segment.sum()
    samples_7 = np.empty((num_samples, 7))
    sample_idx = 0
    for i in range(len(control_points_quat) - 1):
        start_pos, start_xyzw = control_points_quat[i, :3], control_points_quat[i, 3:]
        end_pos, end_xyzw = control_points_quat[i+1, :3], control_points_quat[i+1, 3:]
        # using proper quaternion slerp interpolation
        poses_7 = np.empty((num_samples_per_segment[i], 7))
        for j in range(num_samples_per_segment[i]):
            alpha = j / (num_samples_per_segment[i] - 1)
            pos = start_pos * (1 - alpha) + end_pos * alpha
            blended_xyzw = T.quat_slerp_jitted(start_xyzw, end_xyzw, alpha)
            pose_7 = np.empty(7)
            pose_7[:3] = pos
            pose_7[3:] = blended_xyzw
            poses_7[j] = pose_7
        samples_7[sample_idx:sample_idx+num_samples_per_segment[i]] = poses_7
        sample_idx += num_samples_per_segment[i]
    assert num_samples >= 2, f'num_samples: {num_samples}'
    return samples_7, num_samples

@njit(cache=True, fastmath=True)
def path_length(samples_homo):
    assert samples_homo.shape[1:] == (4, 4), 'samples_homo must be of shape (N, 4, 4)'
    pos_length = 0
    rot_length = 0
    for i in range(len(samples_homo) - 1):
        pos_length += np.linalg.norm(samples_homo[i, :3, 3] - samples_homo[i+1, :3, 3])
        rot_length += angle_between_rotmat(samples_homo[i, :3, :3], samples_homo[i+1, :3, :3])
    return pos_length, rot_length

# ===============================================
# = others
# ===============================================
def get_callable_grasping_cost_fn(env):
    def get_grasping_cost(keypoint_idx):
        keypoint_object = env.get_object_by_keypoint(keypoint_idx)
        return -env.is_grasping(candidate_obj=keypoint_object) + 1  # return 0 if grasping an object, 1 if not grasping any object
    return get_grasping_cost

def get_config(config_path=None):
    if config_path is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_file_dir, 'configs/config.yaml')
    assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_clock_time(milliseconds=False):
    curr_time = datetime.datetime.now()
    if milliseconds:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}.{curr_time.microsecond // 1000}'
    else:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}'

def angle_between_quats(q1, q2):
    """Angle between two quaternions"""
    return 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), -1, 1))

def filter_points_by_bounds(points, bounds_min, bounds_max, strict=True):
    """
    Filter points by taking only points within workspace bounds.
    """
    assert points.shape[1] == 3, "points must be (N, 3)"
    bounds_min = bounds_min.copy()
    bounds_max = bounds_max.copy()
    if not strict:
        bounds_min[:2] = bounds_min[:2] - 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_max[:2] = bounds_max[:2] + 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_min[2] = bounds_min[2] - 0.1 * (bounds_max[2] - bounds_min[2])
    within_bounds_mask = (
        (points[:, 0] >= bounds_min[0])
        & (points[:, 0] <= bounds_max[0])
        & (points[:, 1] >= bounds_min[1])
        & (points[:, 1] <= bounds_max[1])
        & (points[:, 2] >= bounds_min[2])
        & (points[:, 2] <= bounds_max[2])
    )
    return within_bounds_mask

def print_opt_debug_dict(debug_dict):
    print('\n' + '#' * 40)
    print(f'# Optimization debug info:')
    max_key_length = max(len(str(k)) for k in debug_dict.keys())
    for k, v in debug_dict.items():
        if isinstance(v, int) or isinstance(v, float):
            print(f'# {k:<{max_key_length}}: {v:.05f}')
        elif isinstance(v, list) and all(isinstance(x, int) or isinstance(x, float) for x in v):
            print(f'# {k:<{max_key_length}}: {np.array(v).round(5)}')
        else:
            print(f'# {k:<{max_key_length}}: {v}')
    print('#' * 40 + '\n')

def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    
def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    # for phrase in banned_phrases:
        # assert phrase not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e
    

def load_functions_from_txt(txt_path, function_dict, return_code=False):
    if txt_path is None:
        return []
    # load txt file
    with open(txt_path, 'r') as f:
        functions_text = f.read()
    # execute functions
    gvars_dict = {
        'np': np,
    }  # external library APIs
    gvars_dict.update(function_dict)
    lvars_dict = dict()
    exec_safe(functions_text, gvars=gvars_dict, lvars=lvars_dict)
    if "__doc__" in lvars_dict.keys():
        lvars_dict.pop("__doc__")
    funcs = list(lvars_dict.values())
    if return_code:
        return {"func": funcs, "code": functions_text}
    else:
        return funcs

@njit(cache=True, fastmath=True)
def angle_between_rotmat(P, Q):
    R = np.dot(P, Q.T)
    cos_theta = (np.trace(R)-1)/2
    if cos_theta > 1:
        cos_theta = 1
    elif cos_theta < -1:
        cos_theta = -1
    return np.arccos(cos_theta)

def fit_b_spline(control_points):
    # determine appropriate k
    k = min(3, control_points.shape[0]-1)
    spline = interpolate.splprep(control_points.T, s=0, k=k)
    return spline

def sample_from_spline(spline, num_samples):
    sample_points = np.linspace(0, 1, num_samples)
    if isinstance(spline, RotationSpline):
        samples = spline(sample_points).as_matrix()  # [num_samples, 3, 3]
    else:
        assert isinstance(spline, tuple) and len(spline) == 2, 'spline must be a tuple of (tck, u)'
        tck, u = spline
        samples = interpolate.splev(np.linspace(0, 1, num_samples), tck)  # [spline_dim, num_samples]
        samples = np.array(samples).T  # [num_samples, spline_dim]
    return samples

def linear_interpolate_poses(start_pose, end_pose, num_poses):
    """
    Interpolate between start and end pose.
    """
    assert num_poses >= 2, 'num_poses must be at least 2'
    if start_pose.shape == (6,) and end_pose.shape == (6,):
        start_pos, start_euler = start_pose[:3], start_pose[3:]
        end_pos, end_euler = end_pose[:3], end_pose[3:]
        start_rotmat = T.euler2mat(start_euler)
        end_rotmat = T.euler2mat(end_euler)
    elif start_pose.shape == (4, 4) and end_pose.shape == (4, 4):
        start_pos = start_pose[:3, 3]
        start_rotmat = start_pose[:3, :3]
        end_pos = end_pose[:3, 3]
        end_rotmat = end_pose[:3, :3]
    elif start_pose.shape == (7,) and end_pose.shape == (7,):
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        start_rotmat = T.quat2mat(start_quat)
        end_pos, end_quat = end_pose[:3], end_pose[3:]
        end_rotmat = T.quat2mat(end_quat)
    else:
        raise ValueError('start_pose and end_pose not recognized')
    slerp = Slerp([0, 1], R.from_matrix([start_rotmat, end_rotmat]))
    poses = []
    for i in range(num_poses):
        alpha = i / (num_poses - 1)
        pos = start_pos * (1 - alpha) + end_pos * alpha
        rotmat = slerp(alpha).as_matrix()
        if start_pose.shape == (6,):
            euler = T.mat2euler(rotmat)
            poses.append(np.concatenate([pos, euler]))
        elif start_pose.shape == (4, 4):
            pose = np.eye(4)
            pose[:3, :3] = rotmat
            pose[:3, 3] = pos
            poses.append(pose)
        elif start_pose.shape == (7,):
            quat = T.mat2quat(rotmat)
            pose = np.concatenate([pos, quat])
            poses.append(pose)
    return np.array(poses)

def spline_interpolate_poses(control_points, num_steps):
    """
    Interpolate between through the control points using spline interpolation.
    1. Fit a b-spline through the positional terms of the control points.
    2. Fit a RotationSpline through the rotational terms of the control points.
    3. Sample the b-spline and RotationSpline at num_steps.

    Args:
        control_points: [N, 6] position + euler or [N, 4, 4] pose or [N, 7] position + quat
        num_steps: number of poses to interpolate
    Returns:
        poses: [num_steps, 6] position + euler or [num_steps, 4, 4] pose or [num_steps, 7] position + quat
    """
    assert num_steps >= 2, 'num_steps must be at least 2'
    if isinstance(control_points, list):
        control_points = np.array(control_points)
    if control_points.shape[1] == 6:
        control_points_pos = control_points[:, :3]  # [N, 3]
        control_points_euler = control_points[:, 3:]  # [N, 3]
        control_points_rotmat = []
        for control_point_euler in control_points_euler:
            control_points_rotmat.append(T.euler2mat(control_point_euler))
        control_points_rotmat = np.array(control_points_rotmat)  # [N, 3, 3]
    elif control_points.shape[1] == 4 and control_points.shape[2] == 4:
        control_points_pos = control_points[:, :3, 3]  # [N, 3]
        control_points_rotmat = control_points[:, :3, :3]  # [N, 3, 3]
    elif control_points.shape[1] == 7:
        control_points_pos = control_points[:, :3]
        control_points_rotmat = []
        for control_point_quat in control_points[:, 3:]:
            control_points_rotmat.append(T.quat2mat(control_point_quat))
        control_points_rotmat = np.array(control_points_rotmat)
    else:
        raise ValueError('control_points not recognized')
    # remove the duplicate points (threshold 1e-3)
    diff = np.linalg.norm(np.diff(control_points_pos, axis=0), axis=1)
    mask = diff > 1e-3
    # always keep the first and last points
    mask = np.concatenate([[True], mask[:-1], [True]])
    control_points_pos = control_points_pos[mask]
    control_points_rotmat = control_points_rotmat[mask]
    # fit b-spline through positional terms control points
    pos_spline = fit_b_spline(control_points_pos)
    # fit RotationSpline through rotational terms control points
    times = pos_spline[1]
    rotations = R.from_matrix(control_points_rotmat)
    rot_spline = RotationSpline(times, rotations)
    # sample from the splines
    pos_samples = sample_from_spline(pos_spline, num_steps)  # [num_steps, 3]
    rot_samples = sample_from_spline(rot_spline, num_steps)  # [num_steps, 3, 3]
    if control_points.shape[1] == 6:
        poses = []
        for i in range(num_steps):
            pose = np.concatenate([pos_samples[i], T.mat2euler(rot_samples[i])])
            poses.append(pose)
        poses = np.array(poses)
    elif control_points.shape[1] == 4 and control_points.shape[2] == 4:
        poses = np.empty((num_steps, 4, 4))
        poses[:, :3, :3] = rot_samples
        poses[:, :3, 3] = pos_samples
        poses[:, 3, 3] = 1
    elif control_points.shape[1] == 7:
        poses = np.empty((num_steps, 7))
        for i in range(num_steps):
            quat = T.mat2quat(rot_samples[i])
            pose = np.concatenate([pos_samples[i], quat])
            poses[i] = pose
    return poses

def get_linear_interpolation_steps(start_pose, end_pose, pos_step_size, rot_step_size):
    """
    Given start and end pose, calculate the number of steps to interpolate between them.
    Args:
        start_pose: [6] position + euler or [4, 4] pose or [7] position + quat
        end_pose: [6] position + euler or [4, 4] pose or [7] position + quat
        pos_step_size: position step size
        rot_step_size: rotation step size
    Returns:
        num_path_poses: number of poses to interpolate
    """
    if start_pose.shape == (6,) and end_pose.shape == (6,):
        start_pos, start_euler = start_pose[:3], start_pose[3:]
        end_pos, end_euler = end_pose[:3], end_pose[3:]
        start_rotmat = T.euler2mat(start_euler)
        end_rotmat = T.euler2mat(end_euler)
    elif start_pose.shape == (4, 4) and end_pose.shape == (4, 4):
        start_pos = start_pose[:3, 3]
        start_rotmat = start_pose[:3, :3]
        end_pos = end_pose[:3, 3]
        end_rotmat = end_pose[:3, :3]
    elif start_pose.shape == (7,) and end_pose.shape == (7,):
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        start_rotmat = T.quat2mat(start_quat)
        end_pos, end_quat = end_pose[:3], end_pose[3:]
        end_rotmat = T.quat2mat(end_quat)
    else:
        raise ValueError('start_pose and end_pose not recognized')
    pos_diff = np.linalg.norm(start_pos - end_pos)
    rot_diff = angle_between_rotmat(start_rotmat, end_rotmat)
    pos_num_steps = np.ceil(pos_diff / pos_step_size)
    rot_num_steps = np.ceil(rot_diff / rot_step_size)
    num_path_poses = int(max(pos_num_steps, rot_num_steps))
    num_path_poses = max(num_path_poses, 2)  # at least start and end poses
    return num_path_poses

def farthest_point_sampling(pc, num_points):
    """
    Given a point cloud, sample num_points points that are the farthest apart.
    Use o3d farthest point sampling.
    """
    assert pc.ndim == 2 and pc.shape[1] == 3, "pc must be a (N, 3) numpy array"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    downpcd_farthest = pcd.farthest_point_down_sample(num_points)
    return np.asarray(downpcd_farthest.points)

global ENV
ENV = None

def grasp(name):
    env = ENV
    if name == "":
        return {
            "subgoal_pose": env.get_ee_pose(),
        }
    
    import open3d as o3d
    part_to_pts_dict = env.get_part_to_pts_dict()[-1]
    segm_pts_3d = part_to_pts_dict[name]
    rgbs = env.last_cam_obs[1]['rgb']
    pts_3d = env.last_cam_obs[1]['points']
    pcd_debug = o3d.geometry.PointCloud()
    pcd_debug.points = o3d.utility.Vector3dVector(pts_3d.reshape(-1, 3))
    pcd_debug.colors = o3d.utility.Vector3dVector(rgbs.reshape(-1, 3) / 255.)
    pcs_mean = segm_pts_3d.mean(0)
    segm_pts_3d -= pcs_mean
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(segm_pts_3d)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((segm_pts_3d.shape[0], 3)))
    o3d.io.write_point_cloud("tmp.pcd", pcd)
    pcd.points = o3d.utility.Vector3dVector(segm_pts_3d + pcs_mean)
    o3d.io.write_point_cloud("tmp.ply", pcd)
    import subprocess
    grasp_cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../gpd/cfg/eigen_params_mw.cfg")
    grasp_bin_path = "detect_grasps"
    output = subprocess.check_output(['{}'.format(grasp_bin_path), '{}'.format(grasp_cfg_path), "tmp.pcd"])
    app_strs = str(output).split("Approach")[1:]
    approaches = []
    for app_str in app_strs:
        app_str = app_str.strip().split(':')[1].strip()
        app_vec =  app_str.split("\\n")
        app_vec = np.array([float(app_vec[0]), float(app_vec[1]), float(app_vec[2])])
        approaches.append(app_vec)
    approaches = np.stack(approaches, axis=0)
    pos_str = app_strs[-1]
    pos_strs = pos_str.split("Position")[1:]
    positions = []
    for pos_str in pos_strs:
        pos_str = pos_str.strip().split(':')[1].strip()
        pos_vec =  pos_str.split("\\n")
        pos_vec = np.array([float(pos_vec[0]), float(pos_vec[1]), float(pos_vec[2])])
        positions.append(pos_vec)
    positions = np.stack(positions, axis=0)

    binormal_str = pos_strs[-1]
    binormal_strs = binormal_str.split("Binormal")[1:]
    binormals = []
    for binormal_str in binormal_strs:
        binormal_str = binormal_str.strip().split(':')[1].strip()
        binormal_vec =  binormal_str.split("\\n")
        binormal_vec = np.array([float(binormal_vec[0]), float(binormal_vec[1]), float(binormal_vec[2])])
        binormals.append(binormal_vec)
    binormals = np.stack(binormals, axis=0)

    approach0 = env.robot.arm_links['0'][-1].get_position() - env.robot.arm_links['0'][-2].get_position()
    approach0 /= np.linalg.norm(approach0)
    binormal0 = env.robot.finger_links['0'][0].get_position() - env.robot.finger_links['0'][1].get_position()
    binormal0 /= np.linalg.norm(binormal0)
    draw_arrow(pcd_debug, start=env.robot.finger_links['0'][1].get_position(), end=env.robot.finger_links['0'][0].get_position(), color=np.array([0, 0.5, 0]))
    draw_arrow(pcd_debug, start=env.robot.arm_links['0'][-2].get_position(), end= env.robot.arm_links['0'][-1].get_position(), color=np.array([0.5, 0.5, 0]))

    vec0 = binormal0 - approach0
    starts = positions + pcs_mean
    ik_solvable = []
    target_quats = []
    transform_mats = []
    for i in range(len(approaches)):
        approach = approaches[i]
        binormal = binormals[i]
        start = starts[i]
        vec = binormal - approach
        from scipy.spatial.transform import Rotation as R
        def find_rigid_transformation(source_points, target_points):

            # Step 1: Calculate the centroids of both point sets
            centroid_source = np.mean(source_points, axis=0)
            centroid_target = np.mean(target_points, axis=0)

            # Step 2: Center the points around the origin
            centered_source = source_points - centroid_source
            centered_target = target_points - centroid_target

            # Step 3: Compute the covariance matrix
            H = np.dot(centered_source.T, centered_target)

            # Step 4: Perform Singular Value Decomposition (SVD)
            U, S, Vt = np.linalg.svd(H)

            # Step 5: Compute the optimal rotation matrix
            R = np.dot(Vt.T, U.T)

            # Ensure a right-handed coordinate system (if det(R) == -1, flip sign)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)

            # Step 6: Compute the translation vector
            t = centroid_target.T - np.dot(R, centroid_source.T)
            return R
        source_points = np.stack([approach0, binormal0, np.array([0,0,0])], axis=0)
        target_points = np.stack([approach, binormal, np.array([0,0,0])], axis=0)
        # transform_mat = find_rigid_transformation(source_points, target_points)
        transform_mat0 = R.from_quat(env.robot.eef_links['0'].get_orientation()).as_matrix()
        transform_mat =  cv2.estimateAffine3D(source_points, target_points, force_rotation=True)[0][:3, :3]
        transform_mats.append(transform_mat)
        ## TODO:
        # transform_mat = np.eye(3)
        mat = transform_mat @ transform_mat0
        target_quat = R.from_matrix(mat).as_quat()
        target_quats.append(target_quat)
        control_point_homo = T.convert_pose_euler2mat(T.convert_pose_quat2euler(np.concatenate([start, target_quat], axis=-1)))
        max_iterations = 20
        initial_joint_pos = ENV.get_arm_joint_postions()
        from ik_solver import IKSolver
        ik_solver = IKSolver(
            robot_description_path=ENV.robot.robot_arm_descriptor_yamls[ENV.robot.default_arm],
            robot_urdf_path=ENV.robot.urdf_path,
            eef_name=ENV.robot.eef_link_names[ENV.robot.default_arm],
            reset_joint_pos=ENV.reset_joint_pos,
            world2robot_homo=ENV.world2robot_homo,
        )
        ik_result = ik_solver.solve(
            control_point_homo,
            max_iterations=max_iterations,
            initial_joint_pos=initial_joint_pos,
        )
        ik_solvable.append(ik_result.success)
    target_quats = np.stack(target_quats, axis=0)
    ik_solvable = np.array(ik_solvable)
    transform_mats = np.stack(transform_mats)
    approaches, binormals, starts = approaches[ik_solvable], binormals[ik_solvable], starts[ik_solvable]
    target_quats = target_quats[ik_solvable]
    transform_mats = transform_mats[ik_solvable]
    ## TODO: selection preference: -z
    # ind = (approaches[:, 0]).argmin()
    # ind = approaches[:, 1].argmin()
    # ind = (approaches * np.array([-1, 0, 0])).sum(-1).argmax() 
    ## TODO: for fridge
    ind = ((approaches * np.array([-1, 0, 0])).sum(-1) + binormal * np.array([0, 1, 0])).argmax() 

    approach = approaches[ind]
    binormal = binormals[ind]
    target_quat = target_quats[ind]
    transform_mat = transform_mats[ind]
    binormal = binormal / np.linalg.norm(binormal)
    approach = approach / np.linalg.norm(approach)
    start = starts[ind] 
    end = start + approach / 5.
    pcd_debug = draw_arrow(pcd_debug, start, end)
    end = start + binormal / 5.
    pcd_debug = draw_arrow(pcd_debug, start, end, color=np.array([0., 1., 0]))
    approach = approach0 @ transform_mat
    binormal = binormal0 @ transform_mat
    end = start + binormal / 5.
    pcd_debug = draw_arrow(pcd_debug, start, end, color=np.array([0., 0.3, 0]))
    end = start + approach / 5.
    pcd_debug = draw_arrow(pcd_debug, start, end, color=np.array([0.3, 0.3, 0]))

    o3d.io.write_point_cloud('grasp_vis.ply', pcd_debug)
    target_position = start
    subgoal_pose = np.concatenate([target_position, target_quat])
    np.save("grasp_pose_cache.npy", subgoal_pose)
    return {
        "subgoal_pose": np.concatenate([target_position, target_quat]),
    }
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_prompt(prompt_path, prompt):
    prompts = prompt.split("<IMAGE>")
    contents = []
    for i in range(len(prompts)):
        a_prompt = prompts[i]
        a_prompt_text = a_prompt.split("</IMAGE>")[-1]
        a_prompt_img_path = os.path.join(prompt_path, a_prompt.split("</IMAGE>")[0])
        if i > 0:
            if os.path.exists(a_prompt_img_path):
                base64_image = encode_image(a_prompt_img_path)
                contents.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                )
        contents.append(
            {
                "type": "text",
                "text": a_prompt_text,
            }
        )
    return contents
    
def draw_arrow(pcd, start, end, color=np.array([1, 1, 0])):
    pts = np.linspace(start, end, num=50)
    pcs = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    pcs = np.concatenate([pcs, pts], axis=0)
    cols = np.concatenate([cols, np.ones((len(pts), 3)) * color])
    pcd.points = o3d.utility.Vector3dVector(pcs)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd
    
from scipy.spatial.transform import Rotation as R

def grasp_all_candidates(name):
    env = ENV
    if name == "":
        return {
            "subgoal_poses": env.get_ee_pose()[None, ...],
        }
    
    import open3d as o3d
    part_to_pts_dict = copy.deepcopy(env.get_part_to_pts_dict()[-1])
    segm_pts_3d = part_to_pts_dict[name]
    rgbs = env.last_cam_obs[1]['rgb']
    pts_3d = env.last_cam_obs[1]['points']
    pcd_debug = o3d.geometry.PointCloud()
    pcd_debug.points = o3d.utility.Vector3dVector(pts_3d.reshape(-1, 3))
    pcd_debug.colors = o3d.utility.Vector3dVector(rgbs.reshape(-1, 3) / 255.)

    pcs_mean = segm_pts_3d.mean(0)
    segm_pts_3d -= pcs_mean
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(segm_pts_3d)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((segm_pts_3d.shape[0], 3)))
    o3d.io.write_point_cloud("tmp.pcd", pcd)
    pcd.points = o3d.utility.Vector3dVector(segm_pts_3d + pcs_mean)
    o3d.io.write_point_cloud("tmp.ply", pcd)
    import subprocess
    grasp_cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../gpd/cfg/eigen_params.cfg")
    grasp_bin_path = "detect_grasps"
    output = subprocess.check_output(['{}'.format(grasp_bin_path), '{}'.format(grasp_cfg_path), "tmp.pcd"])
    app_strs = str(output).split("Approach")[1:]
    approaches = []
    for app_str in app_strs:
        app_str = app_str.strip().split(':')[1].strip()
        app_vec =  app_str.split("\\n")
        app_vec = np.array([float(app_vec[0]), float(app_vec[1]), float(app_vec[2])])
        approaches.append(app_vec)
    approaches = np.stack(approaches, axis=0)
    pos_str = app_strs[-1]
    pos_strs = pos_str.split("Position")[1:]
    positions = []
    for pos_str in pos_strs:
        pos_str = pos_str.strip().split(':')[1].strip()
        pos_vec =  pos_str.split("\\n")
        pos_vec = np.array([float(pos_vec[0]), float(pos_vec[1]), float(pos_vec[2])])
        positions.append(pos_vec)
    positions = np.stack(positions, axis=0)

    binormal_str = pos_strs[-1]
    binormal_strs = binormal_str.split("Binormal")[1:]
    binormals = []
    for binormal_str in binormal_strs:
        binormal_str = binormal_str.strip().split(':')[1].strip()
        binormal_vec =  binormal_str.split("\\n")
        binormal_vec = np.array([float(binormal_vec[0]), float(binormal_vec[1]), float(binormal_vec[2])])
        binormals.append(binormal_vec)
    binormals = np.stack(binormals, axis=0)

    approach0 = env.APPROACH0.astype(np.float32)
    approach0 /= np.linalg.norm(approach0)
    binormal0 = env.BINORMAL0.astype(np.float32)
    binormal0 /= np.linalg.norm(binormal0)

    approaches = -approaches
    starts = positions + pcs_mean + 0.03 * approaches
    target_quats = []
    transform_mats = []

    pcd_debug = o3d.geometry.PointCloud()

    for i in range(len(approaches)):
        approach = approaches[i]
        binormal = binormals[i]

        source_points = np.stack([approach0, binormal0, np.array([0,0,0])], axis=0)
        target_points = np.stack([approach, binormal, np.array([0,0,0])], axis=0)
        transform_mat0 = np.identity(3)
        transform_mat =  cv2.estimateAffine3D(source_points, target_points, force_rotation=True)[0][:3, :3]
        transform_mats.append(transform_mat)
        mat = transform_mat @ transform_mat0
        target_quat = R.from_matrix(mat).as_quat()
        target_quats.append(target_quat)

    target_quats = np.stack(target_quats, axis=0)
    target_positions = starts
    subgoal_poses = np.concatenate([target_positions, target_quats], axis=-1)
    return {
        "subgoal_poses": subgoal_poses,
        "pcd_debug": pcd_debug
    }
# Function to encode the image
def encode_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image.copy(), (image.shape[1] // 2, image.shape[0] // 2))
    max_width, max_height = 100, 100
    f1 = max_width / image.shape[1]
    f2 = max_height / image.shape[0]
    f = min(f1, f2)  # resizing factor
    dim = (int(image.shape[1] * f), int(image.shape[0] * f))
    image = cv2.resize(image.copy(), dim)
    cv2.imwrite('segm_tmp.png', image)
    with open('segm_tmp.png', "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_prompt(prompt_path, prompt):
    prompts = prompt.split("<IMAGE>")
    contents = []
    for i in range(len(prompts)):
        a_prompt = prompts[i]
        a_prompt_text = a_prompt.split("</IMAGE>")[-1]
        a_prompt_img_path = os.path.join(prompt_path, a_prompt.split("</IMAGE>")[0])
        if i > 0:
            if os.path.exists(a_prompt_img_path):
                base64_image = encode_image(a_prompt_img_path)
                contents.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                )
        contents.append(
            {
                "type": "text",
                "text": a_prompt_text,
            }
        )
    return contents
    
def _center_geometry(ee_pose, part_to_pts_dict_3d_original, moving_part_names, pos_only=False):
    part_to_pts_dict_3d = copy.deepcopy(part_to_pts_dict_3d_original)
    if len(ee_pose[3:]) == 3:
        ee_pose = np.concatenate([ee_pose[:3], T.euler2quat(ee_pose[3:])])
    ee_pose_homo = T.pose2mat([ee_pose[:3], ee_pose[3:]])
    centering_transform = np.linalg.inv(ee_pose_homo)
    part_to_pts_dict2_3d = {}
    for key in part_to_pts_dict_3d[-1].keys():
        if key in moving_part_names:
            if not pos_only:
                part_to_pts_dict2_3d[key] = np.dot(part_to_pts_dict_3d[-1][key], centering_transform[:3, :3].T) + centering_transform[:3, 3]
            else:
                part_to_pts_dict2_3d[key] = part_to_pts_dict_3d[-1][key] + centering_transform[:3, 3]
        else:
            part_to_pts_dict2_3d[key] = part_to_pts_dict_3d[-1][key]
    part_to_pts_dict_3d[-1] = part_to_pts_dict2_3d
    return part_to_pts_dict_3d
from path_solver import get_collision_cost

def voxelize_pcs(pcs):
    pcs = pcs[np.logical_not(np.isnan(pcs).any(-1))]
    pcs = np.unique(np.round(pcs / 0.04).astype(np.int64), axis=0)
    return pcs

def select_grasp_with_constraints(subgoal_poses, constraints, occlusion_func=None):
    env = ENV
    moving_part_names = env.get_moving_part_names()
    start_pose = env.get_ee_pose()
    part_to_pts_dict_3d_centered = _center_geometry(start_pose, env.part_to_pts_dict, moving_part_names)
    costs = []
    for subgoal_pose in subgoal_poses:
        subgoal_pose_homo = T.pose2mat([subgoal_pose[:3], subgoal_pose[3:]])
        transformed_part_to_pts_dict_3d = transform_geometry(subgoal_pose_homo, part_to_pts_dict_3d_centered, moving_part_names)
        env.part_to_pts_dict_simulation = copy.deepcopy(transformed_part_to_pts_dict_3d)
        video_tmp_original = env.video_tmp
        env.video_tmp = []
        cost = 0.
        for constraint in constraints:
            cost += constraint()
        if occlusion_func is not None:
            collision_cost = get_collision_cost(transformed_part_to_pts_dict_3d, moving_part_names, occlusion_func)
        else:
            collision_cost = 0.
        approach0 = env.robot.APPROACH0
        approach_vec = approach0 @ subgoal_pose_homo[:3, :3].T
        # forbidden_approach_vec = np.array([1, 0, 0])
        # if np.dot(forbidden_approach_vec, approach_vec) > 0.:
        #     cost += 10000
        # forbidden_approach_vec = np.array([0, 0, 1])
        # if np.dot(forbidden_approach_vec, approach_vec) > 0.5:
        #     cost += 10000

        ## TODO: camera preferences
        cam_vec = np.array([0, 0, 1])
        cam_to_world_rot = env.camera.extrinsic[:3, :3]
        cam_vec = cam_vec @ cam_to_world_rot.T
        gripper_vec = transformed_part_to_pts_dict_3d[-1]['the gripper approach of the robot']
        gripper_vec = gripper_vec[0] - gripper_vec[-1]
        gripper_vec /= np.linalg.norm(gripper_vec)
        cam_aligned = np.dot(cam_vec, gripper_vec)
        if cam_aligned < 0:
            cam_aligned_cost = -cam_aligned
        else:
            cam_aligned_cost = 0.
        cost += 50 * cam_aligned_cost
        ## end TODO
        
        cost += 100 * collision_cost
        env.part_to_pts_dict_simulation = None
        env.video_tmp = video_tmp_original
        costs.append(cost)
    costs = np.stack(costs, axis=0)
    print(costs)
    subgoal_pose = subgoal_poses[np.argmin(costs)]
    return subgoal_pose

from openai import OpenAI
def query_GPT(query):
    env = ENV
    rgb = env.get_cam_obs(use_last=False)[1]['rgb']
    client = OpenAI()
    cv2.imwrite("rgb_obs.png", rgb[:, :, ::-1])
    base64_image = encode_image("rgb_obs.png")
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    contents = []
    contents += [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        },
        {
            "type": "text",
            "text": query,
        }
    ]
    messages.append(
        {
            "role": "user",
            "content": contents
        }
    )
    completion = client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=messages,
        temperature=0.3,
        top_p=0.2,
    )
    reply = completion.choices[0].message.content
    return reply

def query_vlm_model(client, model, messages, temperature, top_p, stream=False):
    if "o1" not in model:
        completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                stream=stream
            )
    else:
        completion = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream
            )
    if not stream:
        reply = completion.choices[0].message.content
        return reply
    else:
        return completion