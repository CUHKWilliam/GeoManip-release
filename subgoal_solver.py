import numpy as np
import time
import copy
from scipy.optimize import dual_annealing, minimize
from scipy.interpolate import RegularGridInterpolator
import transform_utils as T
import utils as utils
from utils import (
    transform_keypoints,
    transform_geometry,
    calculate_collision_cost,
    normalize_vars,
    unnormalize_vars,
    farthest_point_sampling,
    consistency,
)
from scipy.spatial.transform import Rotation as R
hist_cost = 1e10
opt_pose_global = None
from path_solver import get_collision_cost

def objective(opt_vars,
            og_bounds,
            part_to_pts_dict_3d_centered,
            moving_part_names,
            goal_constraints,
            path_constraints,
            init_pose_homo,
            initial_joint_pos,
            reset_joint_pos,
            is_grasp_stage,
            pos_only=False,
            occlusion_func=None,
            curr_ee_pose=None,
            delta=False,
            return_debug_dict=False):
    debug_dict = {}
    # unnormalize variables and do conversion
    opt_pose = unnormalize_vars(opt_vars, og_bounds)
    if delta:
        assert curr_ee_pose is not None
    if curr_ee_pose is not None:
        if len(curr_ee_pose[3:]) == 4:
            rot = R.from_quat(curr_ee_pose[3:]).as_euler("xyz")
        else:
            rot = curr_ee_pose[3:]
        pos = curr_ee_pose[:3]
        curr_ee_pose = np.concatenate([pos, rot])
    if delta:
        opt_pose = opt_pose + curr_ee_pose
    opt_pose_homo = T.pose2mat([opt_pose[:3], T.euler2quat(opt_pose[3:])])
    cost = 0
    if curr_ee_pose is not None:
        pos = curr_ee_pose[:3]
        curr_ee_pose
        delta_cost = opt_pose - curr_ee_pose
        delta_cost[:3] *= 4.
        delta_cost[3:] *= 80.
        delta_cost = np.sum(np.abs(delta_cost))

    cost += delta_cost
    debug_dict['delta_cost'] = delta_cost
    # collision cost
    # if collision_points_centered is not None:
    #     collision_cost = 0.8 * calculate_collision_cost(opt_pose_homo[None], sdf_func, collision_points_centered, 0.10)
    #     debug_dict['collision_cost'] = collision_cost
    #     cost += collision_cost

    # stay close to initial pose
    init_pose_cost = 0.0 * consistency(opt_pose_homo[None], init_pose_homo[None], rot_weight=20.)
    debug_dict['init_pose_cost'] = init_pose_cost
    cost += init_pose_cost
    
    ## TODO: the approach direction limit
    if not pos_only:
        approach0 = utils.ENV.robot.APPROACH0
        approach_vec = approach0 @ opt_pose_homo[:3, :3].T
        # forbidden_approach_vec = np.array([1, 0, 0])
        # if np.dot(forbidden_approach_vec, approach_vec) > 0.9:
        #     cost += 20000
        forbidden_approach_vec = np.array([0, 0, -1])
        if np.dot(forbidden_approach_vec, approach_vec) > 0.5:
            cost += 20000
        # forbidden_approach_vec = np.array([0, 1, 0])
        # if np.dot(forbidden_approach_vec, approach_vec) > 0.7:
        #     cost += 10000


    # goal constraint violation cost
    debug_dict['subgoal_constraint_cost'] = None
    debug_dict['subgoal_violation'] = None
    
    
    if goal_constraints is not None and len(goal_constraints) > 0:
        transformed_part_to_pts_dict_3d = copy.deepcopy(utils.ENV.part_to_pts_dict)
        transformed_part_to_pts_dict_3d_latest = copy.deepcopy(part_to_pts_dict_3d_centered[-1])
        for part_name in transformed_part_to_pts_dict_3d_latest.keys():
            if part_name in moving_part_names:
                part_pts = transformed_part_to_pts_dict_3d_latest[part_name]
                if not pos_only:
                    transformed_part_to_pts_dict_3d_latest[part_name] = np.dot(part_pts, opt_pose_homo[:3, :3].T) + opt_pose_homo[:3, 3]
                else:
                    transformed_part_to_pts_dict_3d_latest[part_name] = part_pts + opt_pose_homo[:3, 3]
        transformed_part_to_pts_dict_3d.append(transformed_part_to_pts_dict_3d_latest)
        utils.ENV.part_to_pts_dict_simulation = copy.deepcopy(transformed_part_to_pts_dict_3d)
        video_tmp_original = utils.ENV.video_tmp
        utils.ENV.video_tmp = []
        subgoal_constraint_cost = 0
        collision_cost = 0
        subgoal_violation = []
        for constraint in goal_constraints:
            violation = constraint()
            subgoal_violation.append(violation)
            if np.isneginf(violation) or np.isposinf(violation) or np.isnan(violation):
                violation = 0.
            subgoal_constraint_cost += np.clip(violation, 0, 20000.)
        subgoal_constraint_cost = 600.0*subgoal_constraint_cost
        

        if occlusion_func is not None:
            collision_cost += get_collision_cost(transformed_part_to_pts_dict_3d, moving_part_names, occlusion_func)
            collision_cost *= 200.

        debug_dict['subgoal_constraint_cost'] = subgoal_constraint_cost
        debug_dict['subgoal_violation'] = subgoal_violation
        debug_dict["collision_cost"] = collision_cost
        cost += collision_cost
        cost += subgoal_constraint_cost
        ## reset part_to_pts_dict_3d
        utils.ENV.part_to_pts_dict_simulation = None
        utils.ENV.video_tmp = video_tmp_original
        
    # path constraint violation cost
    debug_dict['path_violation'] = None
    if path_constraints is not None and len(path_constraints) > 0:
        ## transform part_to_pts_dict_3d
        transformed_part_to_pts_dict_3d = copy.deepcopy(part_to_pts_dict_3d_centered)
        for part_name in transformed_part_to_pts_dict_3d[-1].keys():
            if part_name in moving_part_names:
                part_pts = transformed_part_to_pts_dict_3d[-1][part_name]
                transformed_part_to_pts_dict_3d[-1][part_name] = np.dot(part_pts, opt_pose_homo[:3, :3].T) + opt_pose_homo[:3, 3]
        utils.ENV.part_to_pts_dict_simulation = copy.deepcopy(transformed_part_to_pts_dict_3d)
        video_tmp_original = utils.ENV.video_tmp
        utils.ENV.video_tmp = []
        path_constraint_cost = 0
        path_violation = []
        for constraint in path_constraints:
            violation = constraint()
            path_violation.append(violation)
            path_constraint_cost += np.clip(violation, 0, np.inf)
        path_constraint_cost = 200.0*path_constraint_cost
        ## TODO:
        # print("path_constraint_cost:", path_constraint_cost)
        debug_dict['path_constraint_cost'] = path_constraint_cost
        debug_dict['path_violation'] = path_violation
        cost += path_constraint_cost
        ## reset part_to_pts_dict_3d
        utils.ENV.part_to_pts_dict_simulation = None
        utils.ENV.video_tmp = video_tmp_original

    debug_dict['total_cost'] = cost

    global opt_pose_global, hist_cost
    if cost < hist_cost:
        hist_cost = cost
        opt_pose_global = opt_pose
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(opt_pose_global[:3][None, :])
        # o3d.io.write_point_cloud('debug2.ply', pcd)

    if return_debug_dict:
        return cost, debug_dict
    return cost


class SubgoalSolver:
    def __init__(self, config, reset_joint_pos=None, pos_only=False):
        self.config = config
        self.reset_joint_pos = reset_joint_pos
        self.last_opt_result = None
        # warmup
        # self._warmup()

    # def _warmup(self):
    #     ee_pose = np.array([0.0, 0.0, 0.0, 0, 0, 0, 1])
    #     keypoints = np.random.rand(10, 3)
    #     keypoint_movable_mask = np.random.rand(10) > 0.5
    #     goal_constraints = []
    #     path_constraints = []
    #     sdf_voxels = np.zeros((10, 10, 10))
    #     collision_points = np.random.rand(100, 3)
    #     self.solve(ee_pose, part_to_pts_dict_3d, moving_part_names, goal_constraints, path_constraints, True, None, from_scratch=True)
    #     self.last_opt_result = None

    def _setup_sdf(self, sdf_voxels):
        # create callable sdf function with interpolation
        x = np.linspace(self.config['bounds_min'][0], self.config['bounds_max'][0], sdf_voxels.shape[0])
        y = np.linspace(self.config['bounds_min'][1], self.config['bounds_max'][1], sdf_voxels.shape[1])
        z = np.linspace(self.config['bounds_min'][2], self.config['bounds_max'][2], sdf_voxels.shape[2])
        sdf_func = RegularGridInterpolator((x, y, z), sdf_voxels, bounds_error=False, fill_value=0)
        return sdf_func

    def _check_opt_result(self, opt_result, debug_dict):
        # accept the opt_result if it's only terminated due to iteration limit
        if (not opt_result.success and ('maximum' in opt_result.message.lower() or 'iteration' in opt_result.message.lower() or 'not necessarily' in opt_result.message.lower())):
            opt_result.success = True
        elif not opt_result.success:
            opt_result.message += '; invalid solution'
        # check whether goal constraints are satisfied
        if debug_dict['subgoal_violation'] is not None:
            goal_constraints_results = np.array(debug_dict['subgoal_violation'])
            opt_result.message += f'; goal_constraints_results: {goal_constraints_results} (higher is worse)'
            goal_constraints_satisfied = all([violation <= self.config['constraint_tolerance'] for violation in goal_constraints_results])
            if not goal_constraints_satisfied:
                opt_result.success = False
                opt_result.message += f'; goal not satisfied'
        # check whether path constraints are satisfied
        if debug_dict['path_violation'] is not None:
            path_constraints_results = np.array(debug_dict['path_violation'])
            opt_result.message += f'; path_constraints_results: {path_constraints_results}'
            path_constraints_satisfied = all([violation <= self.config['constraint_tolerance'] for violation in path_constraints_results])
            if not path_constraints_satisfied:
                opt_result.success = False
                opt_result.message += f'; path not satisfied'
        # check whether ik is feasible
        if 'ik_feasible' in debug_dict and not debug_dict['ik_feasible']:
            opt_result.success = False
            opt_result.message += f'; ik not feasible'
        return opt_result
    
    def _center_collision_points_and_keypoints(self, ee_pose_homo, collision_points, keypoints, keypoint_movable_mask):
        centering_transform = np.linalg.inv(ee_pose_homo)
        collision_points_centered = np.dot(collision_points, centering_transform[:3, :3].T) + centering_transform[:3, 3]
        keypoints_centered = transform_keypoints(centering_transform, keypoints, keypoint_movable_mask)
        return collision_points_centered, keypoints_centered
    

    def solve(self,
            ee_pose,
            part_to_pts_dict_3d,
            moving_part_names,
            goal_constraints,
            path_constraints,
            is_grasp_stage,
            initial_joint_pos,
            from_scratch=False,
            occlusion_func=None,
            pos_only=False,
            delta=False,
            ):
        """
        Args:
            - ee_pose (np.ndarray): [7], [x, y, z, qx, qy, qz, qw] end effector pose.
            - keypoints (np.ndarray): [M, 3] keypoint positions.
            - keypoint_movable_mask (bool): [M] boolean array indicating whether the keypoint is on the grasped object.
            - goal_constraints (List[Callable]): subgoal constraint functions.
            - path_constraints (List[Callable]): path constraint functions.
            - sdf_voxels (np.ndarray): [X, Y, Z] signed distance field of the environment.
            - collision_points (np.ndarray): [N, 3] point cloud of the object.
            - is_grasp_stage (bool): whether the current stage is a grasp stage.
            - initial_joint_pos (np.ndarray): [N] initial joint positions of the robot.
            - from_scratch (bool): whether to start from scratch.
        Returns:
            - result (scipy.optimize.OptimizeResult): optimization result.
            - debug_dict (dict): debug information.
        """
        # downsample collision points        
        # ====================================
        # = setup bounds and initial guess
        # ====================================
        ee_pose = ee_pose.astype(np.float64)
        ee_pose_homo = T.pose2mat([ee_pose[:3], ee_pose[3:]])
        ee_pose_euler = np.concatenate([ee_pose[:3], T.quat2euler(ee_pose[3:])])
        # normalize opt variables to [0, 1]
        if not delta:
            pos_bounds_min = self.config['bounds_min']
            pos_bounds_max = self.config['bounds_max']
            rot_bounds_min = np.array([-2 * np.pi, -2 * np.pi, -2 * np.pi])  # euler angles
            rot_bounds_max = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi])  # euler angles
        else:
            pos_bounds_min = np.array([-0.05, -0.05, -0.05])
            pos_bounds_max = np.array([0.05, 0.05, 0.05])
            rot_bounds_min = np.array([-np.pi / 20., -np.pi / 20., -np.pi / 20.])  # euler angles
            rot_bounds_max = np.array([np.pi / 20., np.pi / 20., np.pi / 20.])  # euler angles
        og_bounds = [(b_min, b_max) for b_min, b_max in zip(np.concatenate([pos_bounds_min, rot_bounds_min]), np.concatenate([pos_bounds_max, rot_bounds_max]))]
        bounds = [(-1, 1)] * len(og_bounds)
        if pos_only:
            for i in range(3, 6):
                og_bounds[i] = (0., 0.0001)
            bounds[3], bounds[4], bounds[5] = (0, 0.0001), (0, 0.0001), (0, 0.0001)

        if not from_scratch and self.last_opt_result is not None:
            init_sol = self.last_opt_result.x
        else:
            init_sol = normalize_vars(ee_pose_euler, og_bounds)  # start from the current pose
            from_scratch = True

        # ====================================
        # = other setup
        # ====================================
        centering_transform = np.linalg.inv(ee_pose_homo)
        part_to_pts_dict_3d_centered = copy.deepcopy(part_to_pts_dict_3d)
        for key in part_to_pts_dict_3d_centered[-1].keys():
            if key in moving_part_names:
                if not pos_only:
                    part_to_pts_dict_3d_centered[-1][key] = np.dot(part_to_pts_dict_3d_centered[-1][key], centering_transform[:3, :3].T) + centering_transform[:3, 3]
                else:
                    part_to_pts_dict_3d_centered[-1][key] = part_to_pts_dict_3d_centered[-1][key] + centering_transform[:3, 3]
        aux_args = (og_bounds,
                    part_to_pts_dict_3d_centered,
                    moving_part_names,
                    goal_constraints,
                    path_constraints,
                    ee_pose_homo,
                    initial_joint_pos,
                    self.reset_joint_pos,
                    is_grasp_stage,
                    pos_only,
                    occlusion_func,
                    ee_pose,
                    delta)

        # ====================================
        # = solve optimization
        # ====================================
        start = time.time()
        # use global optimization for the first iteration
        global hist_cost
        hist_cost = 1e10
        if from_scratch:
            opt_result = dual_annealing(
                func=objective,
                bounds=bounds,
                args=aux_args,
                maxfun=self.config['sampling_maxfun'],
                x0=init_sol,
                no_local_search=False,
                minimizer_kwargs={
                    'method': 'SLSQP',
                    'options': self.config['minimizer_options'],
                },
            )
        # use gradient-based local optimization for the following iterations
        else:
            opt_result = minimize(
                fun=objective,
                x0=init_sol,
                args=aux_args,
                bounds=bounds,
                method='SLSQP',
                options=self.config['minimizer_options'],
            )
        solve_time = time.time() - start

        # ====================================
        # = post-process opt_result
        # ====================================
        if isinstance(opt_result.message, list):
            opt_result.message = opt_result.message[0]
        # rerun to get debug info
        _, debug_dict = objective(opt_result.x, *aux_args, return_debug_dict=True)
        debug_dict['sol'] =  opt_result.x
        debug_dict['msg'] = opt_result.message
        debug_dict['solve_time'] = solve_time
        debug_dict['from_scratch'] = from_scratch
        debug_dict['type'] = 'subgoal_solver'
        # unnormailze
        sol = opt_pose_global
        sol = np.concatenate([sol[:3], T.euler2quat(sol[3:])])
        opt_result = self._check_opt_result(opt_result, debug_dict)
        # cache opt_result for future use if successful
        # if opt_result.success:
        self.last_opt_result = copy.deepcopy(opt_result)
        return sol, debug_dict