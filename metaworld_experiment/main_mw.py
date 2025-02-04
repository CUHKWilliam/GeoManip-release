import imageio.v2 as imageio
import numpy as np
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
from metaworld import policies
from tqdm import tqdm
import cv2
import json
import os
import random
import torch
from argparse import ArgumentParser
import imageio
import shutil

from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)
import utils as utils
from constraint_generation import ConstraintGenerator2
import subprocess
import open3d as o3d
from path_solver import PathSolver
from subgoal_solver import SubgoalSolver
import utils as utils
from segment import segment

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def get_task_text(env_name):
    name = " ".join(env_name.split('-')[:-3])
    return name


with open("name2maskid.json", "r") as f:
    name2maskid = json.load(f)

center_grasp = None


def grasp(name):
    env = utils.ENV
    if name.strip() == "":
        return {
            "subgoal_pos": np.concatenate([env.get_endeff_pos(), np.array([0, 0, 0, 1])]),
        } 
    part_to_pts_dict = env.get_part_to_pts_dict(name2maskid)[-1]
    segm_pts_3d = part_to_pts_dict[name]
    rgbs = env.last_obs['rgb']
    pts_3d = env.last_obs['points']

    import open3d as o3d

    pcd_debug = o3d.geometry.PointCloud()
    pcd_debug.points = o3d.utility.Vector3dVector(pts_3d.reshape(-1, 3))
    pcd_debug.colors = o3d.utility.Vector3dVector(rgbs.reshape(-1, 3) / 255.)
    
    pcs_mean = segm_pts_3d.mean(0)
    global center_grasp
    center_grasp = pcs_mean
    target_position = pcs_mean + np.array([0., 0., 0.1])
    return {
        "subgoal_pos": np.concatenate([target_position, np.array([0, 0, 0, 1])]),
    }

def grasp2(name):
    env = utils.ENV
    part_to_pts_dict = env.get_part_to_pts_dict(name2maskid)[-1]
    segm_pts_3d = part_to_pts_dict[name]
    rgbs = env.last_obs['rgb']
    pts_3d = env.last_obs['points']

    import open3d as o3d

    pcd_debug = o3d.geometry.PointCloud()
    pcd_debug.points = o3d.utility.Vector3dVector(pts_3d.reshape(-1, 3))
    pcd_debug.colors = o3d.utility.Vector3dVector(rgbs.reshape(-1, 3) / 255.)
    
    pcs_mean = segm_pts_3d.mean(0)
    segm_pts_3d -= pcs_mean
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(segm_pts_3d)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((segm_pts_3d.shape[0], 3)))
    o3d.io.write_point_cloud("tmp.pcd", pcd)
    import subprocess
    grasp_cfg_path = "/data/wltang/omnigibson/datasets/gpd/cfg/eigen_params.cfg"
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

    starts = positions + pcs_mean
    ## TODO: selection preference: -z
    ind = (approaches[:, 2] + np.abs(approaches[:, 0])).argmin()
    ## TODO:
    global center_grasp
    ## TODO: compensate for the offset
    center_grasp = starts[ind]
    start = starts[ind] + np.array([-0.06, -0.04, 0.04])
    # pcd = draw_arrow(pcd, start=start, end=start + np.array([0, 0, -0.05]))
    # o3d.io.write_point_cloud('grasp_vis.ply', pcd_debug)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(start[None, :])
    o3d.io.write_point_cloud('debug2.ply', pcd)
    target_position = start
    return {
        "subgoal_pos": np.concatenate([target_position, np.array([0, 0, 0, 1])]),
    }


def release():
    pass

def mask_to_pc():
    pass

def add_to_visualize_buffer(visualize_buffer, visualize_points, visualize_colors):
    assert visualize_points.shape[0] == visualize_colors.shape[0], f'got {visualize_points.shape[0]} for points and {visualize_colors.shape[0]} for colors'
    if len(visualize_points) == 0:
        return
    assert visualize_points.shape[1] == 3
    assert visualize_colors.shape[1] == 3
    # assert visualize_colors.max() <= 1.0 and visualize_colors.min() >= 0.0
    visualize_buffer["points"].append(visualize_points)
    visualize_buffer["colors"].append(visualize_colors)

from utils import filter_points_by_bounds, batch_transform_points
import transform_utils as T

class Visualizer:
    def __init__(self,  env):
        self.env = env
        self.color = np.array([0.05, 0.55, 0.26])
        self.world2viewer = np.array([
            [0.3788, 0.3569, -0.8539, 0.0],
            [0.9198, -0.0429, 0.3901, 0.0],
            [-0.1026, 0.9332, 0.3445, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]).T

    def show_img(self, rgb):
        cv2.imshow('img', rgb[..., ::-1])
        cv2.waitKey(0)
        print('showing image, click on the window and press "ESC" to close and continue')
        cv2.destroyAllWindows()
    
    def show_pointcloud(self, points, colors, save=None):
        # transform to viewer frame
        # points = np.dot(points, self.world2viewer[:3, :3].T) + self.world2viewer[:3, 3]
        # clip color to [0, 1]
        colors = np.clip(colors, 0.0, 1.0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))  # float64 is a lot faster than float32 when added to o3d later
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        o3d.io.write_point_cloud(save, pcd)

    def _get_scene_points_and_colors(self, camera, resolution):
        # scene
        obs = self.env.get_obs(camera, resolution=resolution, body_invisible=True)
        scene_points = []
        scene_colors = []
        cam_points = obs['points'].reshape(-1, 3)
        cam_colors = obs['rgb'].reshape(-1, 3) / 255.0
        # clip to workspace
        within_workspace_mask = filter_points_by_bounds(cam_points, np.array([-1, -1, -1]), np.array([1, 1, 1]), strict=False)
        cam_points = cam_points[within_workspace_mask]
        cam_colors = cam_colors[within_workspace_mask]
        scene_points.append(cam_points)
        scene_colors.append(cam_colors)
        scene_points = np.concatenate(scene_points, axis=0)
        scene_colors = np.concatenate(scene_colors, axis=0)
        return scene_points, scene_colors

    def visualize_subgoal(self, subgoal_pose, part_to_pts_3d_dict, moving_part_names, camera, resolution):
        if len(moving_part_names) == 0:
            return
        visualize_buffer = {
            "points": [],
            "colors": []
        }
        # scene
        scene_points, scene_colors = self._get_scene_points_and_colors(camera, resolution)
        add_to_visualize_buffer(visualize_buffer, scene_points, scene_colors)
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # subgoal
        part_to_pts_dict = self.env.get_part_to_pts_dict(name2maskid)[-1]
        part_points = []
        for part_name in moving_part_names:
            part_points.append(part_to_pts_dict[part_name])
        part_points = np.concatenate(part_points, axis=0)
        ee_pose = np.concatenate([self.env.get_endeff_pos(), np.array([0,0,0,1])])
        ee_pose_homo = T.convert_pose_quat2mat(ee_pose)
        centering_transform = np.linalg.inv(ee_pose_homo)
        part_points_centered = part_points + centering_transform[:3, 3]
        transformed_part_points = batch_transform_points(part_points_centered, subgoal_pose_homo[None], pos_only=True).reshape(-1, 3)
        part_points_colors = np.array([self.color] * len(part_points))
        add_to_visualize_buffer(visualize_buffer, transformed_part_points, part_points_colors)

        visualize_points = np.concatenate(visualize_buffer["points"], axis=0)
        visualize_colors = np.concatenate(visualize_buffer["colors"], axis=0)
        self.show_pointcloud(visualize_points, visualize_colors, "debug.ply")


    def visualize_path(self, path, camera, resolution):
        visualize_buffer = {
            "points": [],
            "colors": []
        }
        # scene
        scene_points, scene_colors = self._get_scene_points_and_colors(camera, resolution)
        add_to_visualize_buffer(visualize_buffer, scene_points, scene_colors)
        # draw curve based on poses
        for t in range(len(path) - 1):
            start = path[t][:3]
            end = path[t + 1][:3]
            num_interp_points = int(np.linalg.norm(start - end) / 0.0002)
            interp_points = np.linspace(start, end, num_interp_points)
            interp_colors = np.tile([0.0, 0.0, 0.0], (num_interp_points, 1))
            # add a tint of white (the higher the j, the more white)
            whitening_coef = 0.3 + 0.5 * (t / len(path))
            interp_colors = (1 - whitening_coef) * interp_colors + whitening_coef * np.array([1, 1, 1])
            add_to_visualize_buffer(visualize_buffer, interp_points, interp_colors)
        # subsample path with a fixed step size
        step_size = 0.05
        subpath = [path[0]]
        for i in range(1, len(path) - 1):
            dist = np.linalg.norm(np.array(path[i][:3]) - np.array(subpath[-1][:3]))
            if dist > step_size:
                subpath.append(path[i])
        subpath.append(path[-1])
        path = np.array(subpath)
        path_length = path.shape[0]
        # path points
        moving_part_names = self.env.get_moving_part_names()
        if len(moving_part_names) == 0:
            print("empty moving part")
            return
        part_to_pts_dict = self.env.get_part_to_pts_dict(name2maskid)[-1]
        part_points = []
        for part_name in moving_part_names:
            part_points.append(part_to_pts_dict[part_name])
        start_pose = np.concatenate([self.env.get_endeff_pos(), np.array([0, 0, 0, 1])], axis=0)
        part_points = np.concatenate(part_points, axis=0)
        num_points = part_points.shape[0]
        centering_transform = np.linalg.inv(T.convert_pose_quat2mat(start_pose))
        part_points_centered = part_points + centering_transform[:3, 3]
        poses_homo = T.convert_pose_quat2mat(path[:, :7])
        transformed_part_points = batch_transform_points(part_points_centered, poses_homo, pos_only=True).reshape(-1, 3)
        part_points_colors = np.array([self.color] * len(part_points))
        # calculate color based on the timestep
        part_points_colors = np.ones([path_length, num_points, 3]) * self.color[None, None]
        for t in range(path_length):
            whitening_coef = 0.3 + 0.5 * (t / path_length)
            part_points_colors[t] = (1 - whitening_coef) * part_points_colors[t] + whitening_coef * np.array([1, 1, 1])
        part_points_colors = part_points_colors.reshape(-1, 3)
        add_to_visualize_buffer(visualize_buffer, transformed_part_points, part_points_colors)
        # geometry
        # num_keypoints = keypoints.shape[0]
        # color_map = matplotlib.colormaps["gist_rainbow"]
        # keypoints_colors = [color_map(i / num_keypoints)[:3] for i in range(num_keypoints)]
        # for i in range(num_keypoints):
        #     add_to_visualize_buffer(visualize_buffer, nearby_points, nearby_colors)
        # visualize
        visualize_points = np.concatenate(visualize_buffer["points"], axis=0)
        visualize_colors = np.concatenate(visualize_buffer["colors"], axis=0)
        self.show_pointcloud(visualize_points, visualize_colors, "debug.ply")

class Main:
    def __init__(self):
        global_config = get_config(config_path="/data/wltang/omnigibson/datasets/ReKep2/configs/config.yaml")
        self.constraint_generator = ConstraintGenerator2(global_config['constraint_generator'], prompt_template_path="./vlm_query/prompt_template.txt")
        self.part_to_pts_dict = []
        self.path_constraint_state = {}
        self.global_config = global_config
        self.config = global_config['main']
        self.grasp_state = 0

    def move(self, from_xyz, to_xyz, p):
        error = to_xyz - from_xyz
        response = p * error
        return response

    def _execute_grasp_action(self, grasping_object_part):
        if self.env.is_grasping:
            return
        pos = self.env.get_endeff_pos()
        # grasp_pos = pos + np.array([0., 0., -0.1])
        global center_grasp
        if center_grasp is None:
            grasp_pos = self.env.get_endeff_pos()
        else:
            grasp_pos = center_grasp + np.array([0.0, 0.0, -0.1])
        cnt = 0
        while True:
            prev_pos = self.env.get_endeff_pos()
            next_action2 = self.move(prev_pos, to_xyz=grasp_pos[:3], p=5.)
            next_action2 = np.append(next_action2, np.array([-1]))
            obs, reward, done, info = self.env.step(next_action2, self.camera, self.resolution)
            print("error:", np.linalg.norm(prev_pos - grasp_pos[:3]))
            if np.linalg.norm(prev_pos - grasp_pos[:3]) < .01 or cnt > 50:
                break
            cnt += 1
        for _ in range(10):
            obs, reward, done, info = self.env.step(np.append(next_action2 * 0, np.array([1])), self.camera, self.resolution)
        self.env.is_grasping = True
        self.env.grasping_object_part = grasping_object_part
        return

    def _execute(self, rekep_program_dir, seed=None):
        init_pos = np.array([-0.11897422,  0.47456859,  0.5744534])
        for _ in range(30):
            prev_pos = self.env.get_endeff_pos()
            next_action2 = self.move(prev_pos, to_xyz=init_pos, p=5)
            gripper = np.array([-1])
            next_action2 = np.append(next_action2, gripper)
            self.env.step(next_action2, self.camera, self.resolution)
        cv2.imwrite('debug.png', self.env.get_obs(self.camera, resolution=self.resolution, body_invisible=False)['rgb'])
        self.path_solver = PathSolver(self.global_config['path_solver'], None, self.env.reset_joint_pos)
        self.subgoal_solver = SubgoalSolver(self.global_config['subgoal_solver'], None, self.env.reset_joint_pos)
        self.visualizer = Visualizer(self.env)

        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        utils.ENV = self.env
        self.env.register_keypoints(self.program_info['object_to_segment'], rekep_program_dir, self.camera, self.resolution, seed=seed)
        # register keypoints to be tracked
        # load constraints
        self.constraint_fns = dict()
        self.constraint_fns_code = dict()
        functions_dict = {
            "grasp": grasp,
            "release": release,
            "env": self.env,
            "np": np,
            "subprocess": subprocess,
            "o3d": o3d,
            "mask_to_pc": mask_to_pc,
            "segment": segment
        }
        for stage in range(1, self.program_info['num_stage'] + 1):  # stage starts with 1
            stage_dict = dict()
            stage_dict_code = dict()
            for constraint_type in ['subgoal', 'path', 'target']:
                load_path = os.path.join(rekep_program_dir, f'stage_{stage}_{constraint_type}_constraints.txt')
                if not os.path.exists(load_path):
                    func, code = [], []
                else:
                    ret = load_functions_from_txt(load_path, functions_dict, return_code=True) 
                    func, code = ret['func'], ret["code"]
                ## merge the target constraints and the sub-goal constraint
                stage_dict[constraint_type] = func
                stage_dict_code[constraint_type] = code
                if constraint_type == "path":
                    for func in stage_dict[constraint_type]:
                        self.path_constraint_state[str(func)] = 0 # set inactivate
            stage_dict["subgoal"] += stage_dict['target']
            self.constraint_fns[stage] = stage_dict
            self.constraint_fns_code[stage] = stage_dict_code

        # main loop
        self.last_sim_step_counter = -np.inf
        self._update_stage(1)
        while True:
            self.curr_ee_pose = np.concatenate([self.env.get_endeff_pos(), np.array([0,0,0,1])])
            self.curr_joint_pos = self.env.get_env_state()[0].qpos
            if self.stage > 1:
                path_constraints = self.constraint_fns[self.stage]['path']
                for constraints in path_constraints:
                    if self.path_constraint_state[str(constraints)] == 0:
                        continue
                    violation = constraints()
                    if violation > self.config['constraint_tolerance']:
                        backtrack = True
                        break
            
            # ====================================
            # = get optimized plan
            # ====================================
            # try:
            next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
            # except:
                # print("error happens")
                # return False, False
            if next_subgoal is None:
                ## release gripper
                pass
            else:
                # next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
                # import ipdb;ipdb.set_trace()
                ## TODO:
                prev_ee_pos = self.env.get_endeff_pos()
                next_path = np.array([next_subgoal])
                self.first_iter = False
                self.action_queue = next_path.tolist()
                # ====================================
                # = execute
                # ====================================
                # determine if we proceed to the next stage
                count = 0
                while len(self.action_queue) > 0: #  and count < self.config['action_steps_per_iter']:
                    next_action = self.action_queue.pop(0)
                    precise = len(self.action_queue) == 0
                    cnt = 0
                    while True:
                        prev_pos = self.env.get_endeff_pos()
                        next_action2 = self.move(prev_pos, to_xyz=next_action[:3], p=5)
                        if self.env.is_grasping:
                            gripper = np.array([1])
                        else:
                            gripper = np.array([-1])
                        next_action2 = np.append(next_action2, gripper)
                        try:
                            obs, reward, done, info = self.env.step(next_action2, self.camera, self.resolution)
                        except:
                            print("exceeded!")
                            return False, True
                        done = info['success']
                        if done:
                            break
                        if np.linalg.norm(prev_pos - next_action[:3]) < 0.01 or cnt > 50:
                            break
                        cnt += 1
                    count += 1
            cv2.imwrite('debug.png', self.env.video_cache[-1][:, :, ::-1])
            ## TODO: debug
            # import ipdb;ipdb.set_trace()
            
            if len(self.action_queue) == 0 or done:
                if self.grasp_state == 1:
                    self._execute_grasp_action(self.grasping_object_part)
                else:
                    self._execute_release_action()
                # if completed, save video and return
                if self.stage == self.program_info['num_stage']: 
                    from datetime import datetime, timedelta
                    previous_datetime = datetime.now() - timedelta(days=1)
                    filename = previous_datetime.strftime('%Y-%m-%d_%H-%M-%S')
                    save_path = os.path.join(rekep_program_dir, "{}.mp4".format(filename))
                    SAVE = True
                    if SAVE:
                        with imageio.get_writer(save_path, fps=30,) as writer:
                            for image in self.env.video_cache:
                                writer.append_data(image)
                        print(f"{bcolors.OKGREEN}Video saved to {save_path}\n\n{bcolors.ENDC}")
                    return done, False
                # progress to next stage
                self._update_stage(self.stage + 1)
        
    
    def _get_next_subgoal(self, from_scratch):
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        # path_constraints = self.constraint_fns[self.stage]['path']
        path_constraints = []
        code = self.constraint_fns_code[self.stage]['subgoal']
        self.env.update_moving_part_names()
        ## TODO: refine the code later
        try:
            result = subgoal_constraints[0]()
        except:
            result = 0.
        if isinstance(result, dict):
            ## grasping
            subgoal_pose = result['subgoal_pos']
            debug_dict = {}
            self.grasp_state = 1
            self.grasping_object_part = code.split('grasp("')[1].split('")')[0]
        elif result is None:
            self.grasp_state = 0
            subgoal_pose = None
            debug_dict = {}
        else:
            subgoal_pose, debug_dict = self.subgoal_solver.solve(
                                                    self.curr_ee_pose,
                                                    self.env.get_part_to_pts_dict(name2maskid),
                                                    self.env.get_moving_part_names(),
                                                    subgoal_constraints,
                                                    path_constraints,
                                                    False,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch,
                                                    pos_only=True)
        # subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        # if self.is_grasp_stage:
        #     subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self.config['grasp_depth'] / 2.0, 0, 0])
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        if subgoal_pose is not None:
            self.visualizer.visualize_subgoal(subgoal_pose, self.env.get_part_to_pts_dict(name2maskid)[-1], self.env.get_moving_part_names(), self.camera, self.resolution)
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(subgoal_pose[:3][None, :])
            o3d.io.write_point_cloud('debug3.ply', pcd)
        import ipdb;ipdb.set_trace()
        return subgoal_pose

    def _get_next_path(self, next_subgoal, from_scratch):
        path_constraints = self.constraint_fns[self.stage]['path']
        path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                    next_subgoal,
                                                    self.env.get_part_to_pts_dict(name2maskid),
                                                    self.env.get_moving_part_names(),
                                                    path_constraints,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch,
                                                    pos_only=True)
        print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        self.visualizer.visualize_path(processed_path, self.camera, self.resolution)
        # import ipdb;ipdb.set_trace()
        return processed_path

    def _process_path(self, path):
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)

        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = 1
        return ee_action_seq

    def _update_stage(self, stage):
        # update stage
        self.stage = stage
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        # self._update_keypoint_movable_mask()
        self.first_iter = True

    def _execute_release_action(self):
        if not self.env.is_grasping:
            return
        done2 = False
        for _ in range(30):
            obs, reward, done, info = self.env.step(np.append(np.zeros(6,), np.array([-1])), self.camera, self.resolution)
            if done:
                done2 = True
        self.env.is_grasping = False
        return done2
    
    def run(self, args):
        result_root = args.result_root
        instruction = args.instruction
        os.makedirs(result_root, exist_ok=True)

        n_exps = args.n_exps
        resolution = (320, 240)
        self.resolution = resolution
        # cameras = ['corner', 'corner2', 'corner3']
        cameras = ['corner3']

        env_name = args.env_name
        print(env_name)
        benchmark_env = env_dict[env_name]

        succes_rates = []
        for camera in cameras:
            self.camera = camera
            success = 0
            for seed in tqdm(range(n_exps)):
                if seed < 1:
                    continue
                print("====== camera: {}, seed: {} =========".format(camera, seed))
                rekep_program_dir = os.path.join("{}_{}_{}".format(args.rekep_program_dir, camera, seed))
                os.makedirs(rekep_program_dir, exist_ok=True)
                self.env = benchmark_env(seed=seed)
                self.env.reset()
                for _ in  range(10):
                    self.env.sim.forward()
                    self.env.env_name = env_name
                retry = 0
                while True:
                    obs = self.env.get_obs(camera, resolution=resolution, body_invisible=True)
                    rgb = obs['rgb']
                    cv2.imwrite('debug.png', self.env.get_obs(camera, resolution=resolution, body_invisible=False)['rgb'])
                    # import ipdb;ipdb.set_trace()
                    rekep_program_dir = self.constraint_generator.generate(rgb, instruction, rekep_program_dir=rekep_program_dir, hint=args.hint, seed=seed)
                    print(f'{bcolors.HEADER}Constraints generated{bcolors.ENDC}')
                    done, exceed  = self._execute(rekep_program_dir)
                    if done:
                        success += 1
                        print("success")
                        break
                    else:
                        print("fail")
                        print("retry {}".format(retry))
                        retry += 1
                        import ipdb;ipdb.set_trace()
                        ## TODO:
                        shutil.rmtree(rekep_program_dir)
                        os.makedirs(rekep_program_dir, exist_ok=True)
                        if exceed:
                            break

            success_rate = success / n_exps
            succes_rates.append(success_rate)
        print(f"Success rates for {env_name}:\n", succes_rates)

if __name__ == "__main__":
    parser = ArgumentParser()

    ## task open the safe box
    # parser.add_argument("--env_name", type=str, default="door-open-v2-goal-observable")
    # parser.add_argument("--instruction", type=str, default="grasp the red handle and open the box")
    # parser.add_argument("--rekep_program_dir", type=str, default="/data/wltang/omnigibson/datasets/ReKep2/AVDC_experiments/experiment/vlm_query/open_safe_boxs")
    # parser.add_argument("--hint", type=str, default="Grasp the red handle on the box first.")

    ## task turn faucet
    # parser.add_argument("--env_name", type=str, default="faucet-open-v2-goal-observable")
    # parser.add_argument("--instruction", type=str, default="turn on the red faucet")
    # parser.add_argument("--rekep_program_dir", type=str, default="/data/wltang/omnigibson/datasets/ReKep2/AVDC_experiments/experiment/vlm_query/faucet_on")
    # parser.add_argument("--hint", type=str, default="rotate the center of red faucet around the axis by at least 90 degrees")


    ## task basketball
    parser.add_argument("--env_name", type=str, default="basketball-v2-goal-observable")
    parser.add_argument("--instruction", type=str, default="put the basketball onto the hoop")
    parser.add_argument("--rekep_program_dir", type=str, default="/data/wltang/omnigibson/datasets/ReKep2/AVDC_experiments/experiment/vlm_query/basketball")
    parser.add_argument("--hint", type=str, default="Lift the ball vertically, move over the hoop, then move down right onto the hoop.")

    # parser.add_argument("--env_name", type=str, default="assembly-v2-goal-observable")
    # parser.add_argument("--instruction", type=str, default="put the round ring into the red stick")
    # parser.add_argument("--rekep_program_dir", type=str, default="/data/wltang/omnigibson/datasets/ReKep2/AVDC_experiments/experiment/vlm_query/assembly")
    # parser.add_argument("--hint", type=str, default="grasp the green handle of the round ring and put the hole into the red stick")

    # parser.add_argument("--env_name", type=str, default="shelf-place-v2-goal-observable")
    # parser.add_argument("--instruction", type=str, default="put the blue cube onto the middle stack of the shelf")
    # parser.add_argument("--rekep_program_dir", type=str, default="/data/wltang/omnigibson/datasets/ReKep2/AVDC_experiments/experiment/vlm_query/shelf-place")
    # parser.add_argument("--hint", type=str, default="grasp the blue cube and lift it vertically before moving to the middle stack of the shelf.")

    # parser.add_argument("--env_name", type=str, default="hammer-v2-goal-observable")
    # parser.add_argument("--instruction", type=str, default="use the hammer to smash the nail")
    # parser.add_argument("--rekep_program_dir", type=str, default="/data/wltang/omnigibson/datasets/ReKep2/AVDC_experiments/experiment/vlm_query/hammer")
    # parser.add_argument("--hint", type=str, default="lift the hammer to the same height to the nail first before smashing into it.")

    # parser.add_argument("--env_name", type=str, default="handle-press-v2-goal-observable")
    # parser.add_argument("--instruction", type=str, default="press the red handle")
    # parser.add_argument("--rekep_program_dir", type=str, default="/data/wltang/omnigibson/datasets/ReKep2/AVDC_experiments/experiment/vlm_query/handle")
    # parser.add_argument("--hint", type=str, default="grasp the red handle and press it down for 20 cm")

    # parser.add_argument("--env_name", type=str, default="button-press-topdown-v2-goal-observable")
    # parser.add_argument("--instruction", type=str, default="press the red button from top-down")
    # parser.add_argument("--rekep_program_dir", type=str, default="/data/wltang/omnigibson/datasets/ReKep2/AVDC_experiments/experiment/vlm_query/button-topdown")
    # parser.add_argument("--hint", type=str, default="move gripper directly above the red button by 10 cm first and press top-down")

    # parser.add_argument("--env_name", type=str, default="button-press-v2-goal-observable")
    # parser.add_argument("--instruction", type=str, default="press the red button from its side")
    # parser.add_argument("--rekep_program_dir", type=str, default="/data/wltang/omnigibson/datasets/ReKep2/AVDC_experiments/experiment/vlm_query/button")
    # parser.add_argument("--hint", type=str, default="when pressing, moves toward the button by at least 50 cm")

    # parser.add_argument("--env_name", type=str, default="hammer-v2-goal-observable")
    # parser.add_argument("--instruction", type=str, default="use the hammer to smash the black nail on the side")
    # parser.add_argument("--rekep_program_dir", type=str, default="/data/wltang/omnigibson/datasets/ReKep2/AVDC_experiments/experiment/vlm_query/hammer")
    # parser.add_argument("--hint", type=str, default="move the hammer with the red handle to colinear with the axis of the body of the black nail by 20 cm beforing smashing into it.")


    parser.add_argument("--n_exps", type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="../ckpts/metaworld")
    parser.add_argument("--milestone", type=int, default=24)
    parser.add_argument("--result_root", type=str, default="../results/results_AVDC_full")
    
    args = parser.parse_args()
   

    assert args.env_name in name2maskid.keys()

    main = Main()
    main.run(args)
