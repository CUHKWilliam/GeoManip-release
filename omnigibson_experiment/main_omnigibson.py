import torch
import numpy as np
import json
import os
import argparse
from og_environment import OGEnv 
from constraint_generation import GeometricAndCodeGenerator
from ik_solver import IKSolver
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver
from visualizer import Visualizer
import transform_utils as T
from omnigibson.robots.fetch import Fetch
from utils import (
    bcolors,
    get_config,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    print_opt_debug_dict,
    grasp_all_candidates,
)
import cv2
import ipdb
import subprocess
import open3d as o3d
import utils as utils
from geometry_parser import GeometryParser, get_point_cloud

def release():
    utils.ENV.open_gripper()
    return

def mask_to_pc(mask):
    env = utils.ENV
    env.get_cam_obs()
    pcs = env.last_cam_obs[env.cam_id]['points'][mask]
    return pcs

class Main:
    def __init__(self, scene_file, config_path="./configs/config.yaml", visualize=False, cam_id=1):
        global_config = get_config(config_path=config_path)
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize
        self.geometry_parser = GeometryParser(global_config['segmentation'])
        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # initialize constraint generator
        self.geometric_and_code_generator = GeometricAndCodeGenerator(global_config['constraint_generator'])
        # initialize environment
        self.env = OGEnv(global_config['env'], scene_file, verbose=False)
        # initialize geometry parser
        self.env.geometry_parser = self.geometry_parser
        # initialize solvers
        self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], self.env.reset_joint_pos)
        self.path_solver = PathSolver(global_config['path_solver'], self.env.reset_joint_pos)
        # initialize visualizer
        if self.visualize:
            self.visualizer = Visualizer(global_config['visualizer'], self.env)
        self.env.previous_pose = self.env.get_ee_pose()
        self.grasp_state = 0
        self.path_constraint_state = {}
        self.cam_id = self.env.cam_id = cam_id
        self.set_object_attribute()
        
        
    def set_object_attribute(self, ):
        objs = self.env.og_env.scene.objects
        for obj in objs:
            if "mass" in obj.get_init_info()['args']:
                link_dict = obj.links
                link_dict["base_link"].mass = obj.get_init_info()['args']['mass']
            
    def perform_task(self, instruction, task_dir=None, hint=""):
        self.task_dir = task_dir
        self.env.reset()
        cam_obs = self.env.get_cam_obs()
        rgb = cam_obs[self.cam_id]['rgb']
        # ====================================================================================
        # 1. Task decompositions, flow control generation and geometric constraint generation
        # ====================================================================================
        object_to_segment = self.constraint_generator.task_decomposition_and_geometric_constraint_generation(rgb, instruction, task_dir=task_dir, hint=hint)

        # ====================================================================================
        # 2. Geometry Parser
        # ====================================================================================
        self.env.register_geometry(object_to_segment, task_dir)
        self.register_moving_part_names(grasp=False)

        # ====================================================================================
        # 3. Cost function generation
        # ====================================================================================
        fns_dict = {
            "get_point_cloud": get_point_cloud,
            "grasp": grasp_all_candidates,
            "release": release,
            "env": self.env,
            "np": np,
            "mask_to_pc": mask_to_pc,
        }
        self.constraint_fns, self.constraint_fns_code = self.constraint_generator.cost_fns_generation(fns_dict, task_dir=task_dir)

        # ====================================================================================
        # 4. Execution
        # ====================================================================================
        self.execute()
    
    def execute(self, ):
        # main loop
        self.last_sim_step_counter = -np.inf
        self.update_stage(1)
        while True:
            self.curr_ee_pose = self.env.get_ee_pose()
            self.curr_joint_pos = self.env.get_arm_joint_postions()
            if self.last_sim_step_counter == self.env.step_counter:
                print(f"{bcolors.WARNING}sim did not step forward within last iteration (HINT: adjust action_steps_per_iter to be larger or the pos_threshold to be smaller){bcolors.ENDC}")
            gripper_info, next_subgoal = self.get_next_subgoal(from_scratch=self.first_iter)
            if gripper_info == "release":
                pass
            else:
                next_path = self.get_next_path(next_subgoal, from_scratch=self.first_iter)
                self.first_iter = False
                self.action_queue = next_path.tolist()
                self.last_sim_step_counter = self.env.step_counter
                count = 0
                while len(self.action_queue) > 0: #  and count < self.config['action_steps_per_iter']:
                    next_action = self.action_queue.pop(0)
                    precise = len(self.action_queue) == 0
                    self.env.execute_action(next_action, precise=precise)
                    count += 1
                self.env.previous_pose = self.env.get_ee_pose()
            if len(self.action_queue) == 0:
                if gripper_info == "grasp":
                    self.execute_grasp_action()
                else:
                    self.execute_release_action()
                # if completed, save video and return
                if self.stage == self.program_info['num_stage']: 
                    self.env.sleep(2.0)
                    save_path = self.env.save_video()
                    print(f"{bcolors.OKGREEN}Video saved to {save_path}\n\n{bcolors.ENDC}")
                    return
                # progress to next stage
                self.update_stage(self.stage + 1)

    def register_moving_part_names(self, grasp=True):
        moving_part_names = []
        if grasp:
            code = self.constraint_fns_code[self.stage]['subgoal']
            ## set moving part the part connected to the end-effector
            moving_part_name = code.split('grasp("')[1].split('")')[0]
            moving_part_obj_name = moving_part_name.split("of")[-1].strip()
            for key in self.env.part_to_pts_dict[-1].keys():
                if "axis" in key or "frame" in key:
                    continue
                if key.split("of")[-1].strip() == moving_part_obj_name:
                    moving_part_names.append(key)
        for key in self.env.part_to_pts_dict[-1].keys():
            if "gripper" in key:
                moving_part_names.append(key)
        self.env.moving_part_names = moving_part_names
  
    def get_next_subgoal(self, from_scratch):
        subgoal_cached_path = os.path.join(self.task_dir, "stage_{}_subgoal.npy".format(self.stage))
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        result = subgoal_constraints[0]()
        gripper_info = result['gripper_info']
        if gripper_info == "grasp":
            ## grasping
            subgoal_poses = result['subgoal_poses']
            grasp_constraints = subgoal_constraints[1:]
            subgoal_pose = utils.select_grasp_with_constraints(subgoal_poses, grasp_constraints, self.env.get_occlusion_func())
            debug_dict = {}
        elif gripper_info == "release":
            self.env.moving_part_names = []
            subgoal_pose = None
            debug_dict = {}
        else:
            if os.path.exists(subgoal_cached_path):
                with open(subgoal_cached_path, "rb") as f:
                    subgoal_pose = np.load(f)
                    debug_dict = {}
            else:
                subgoal_pose, debug_dict = self.subgoal_solver.solve(
                                                        self.curr_ee_pose,
                                                        self.env.get_part_to_pts_dict(),
                                                        self.env.get_moving_part_names(),
                                                        subgoal_constraints,
                                                        path_constraints,
                                                        False,
                                                        self.curr_joint_pos,
                                                        from_scratch=from_scratch)
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose, self.env.get_moving_part_names())
        with open(subgoal_cached_path, 'wb') as f:
            np.save(f, subgoal_pose)
        return gripper_info, subgoal_pose

    def get_next_path(self, next_subgoal, from_scratch):
        path_cached_path = os.path.join(self.task_dir, "stage_{}_path.npy".format(self.stage))
        path_constraints = self.constraint_fns[self.stage]['path']
        if os.path.exists(path_cached_path):
            with open(path_cached_path, "rb") as f:
                processed_path = np.load(f)
            debug_dict = {}
        else:
            path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                        next_subgoal,
                                                        self.env.get_part_to_pts_dict(),
                                                        self.env.get_moving_part_names(),
                                                        path_constraints,
                                                        self.curr_joint_pos,
                                                        from_scratch=from_scratch)
            print_opt_debug_dict(debug_dict)
            processed_path = self.process_path(path)
        if self.visualize:
            self.visualizer.visualize_path(processed_path)
            import ipdb;ipdb.set_trace()
        with open(path_cached_path, "wb") as f:
            np.save(f, processed_path)
        return processed_path

    def process_path(self, path):
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
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    def update_stage(self, stage):
        self.stage = stage
        self.action_queue = []
        self.first_iter = True

    def execute_grasp_action(self):
        if self.env.is_grasping:
            return
        pregrasp_pose = self.env.get_ee_pose()
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[:3] += T.quat2mat(pregrasp_pose[3:]) @ np.array([self.config['grasp_depth'], 0, 0])
        grasp_action = np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]])
        self.env.execute_action(grasp_action, precise=True)
        self.env.is_grasping = True
        self.register_moving_part_names(grasp=True)
    
    def execute_release_action(self):
        if not self.env.is_grasping:
            return
        self.env.open_gripper()
        self.register_moving_part_names(grasp=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pen', help='task to perform')
    parser.add_argument('--use_cached_query', action='store_true', help='instead of querying the VLM, use the cached query')
    parser.add_argument('--visualize', action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()
    args.use_cached_query = True
    task_list = {
        'pen': {
            'scene_file': './configs/og_scene_file_pen.json',
            'instruction': 'put the pen perpendicularly into the black cup',
            'task_dir': './vlm_query/pen-4',
            'hint': "",
            "config_path": "./configs/config.yaml",
            },
        'fridge': {
            'scene_file': './configs/og_scene_file_fridge.json',
            'instruction': 'open the fridge',
            'task_dir': './vlm_query/fridge',
            'hint': "",
            "config_path": "./configs/config_fridge.yaml",
        },
        'carrot': {
            'scene_file': './configs/og_scene_file_carrot.json',
            'instruction': 'cut the carrot with the knife',
            'task_dir': './vlm_query/carrot-6',
            'hint': "",
            "config_path": "./configs/config.yaml",
        },
        'computer keyboard': {
            'scene_file': './configs/og_scene_file_computer-keyboard.json',
            'instruction': 'type "hi" on the computer keyboard',
            'task_dir': './vlm_query/computer-keyboard-2',
            'hint': "close the gripper first",
            "cam_id": 2,
            "config_path": "./configs/config.yaml",
        },
    }
    task = task_list['carrot']
    if "cam_id" in task.keys():
        cam_id = task["cam_id"]
    else:
        cam_id = 1
    scene_file = task['scene_file']
    instruction = task['instruction']
    hint = task['hint']
    config_path = task['config_path']
    os.makedirs(task['task_dir'], exist_ok=True)
    main = Main(scene_file, config_path=config_path, visualize=args.visualize, cam_id=cam_id)
    main.perform_task(instruction, task_dir=task['task_dir'] if args.use_cached_query else None, hint=hint)
    