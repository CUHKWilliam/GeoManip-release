import numpy as np
import open3d as o3d
import os
from ...pipeline_base import PipelineBase
from utils.registry import *
import utils.builder as builder
from .utils import *
from scipy.spatial.transform import Rotation as R
from utils.registry import PIPELINES
from .utils import load_cost_functions

@PIPELINES.register_module()
class GeomanipPipeline(PipelineBase):
    def __init__(self, config):
        self.first_iter = True
        self.action_queue = None
        self.grasp_state = 0
        self.grasp_part_name = None
        self.already_grasped = 0
        self.config = config
        self.grasp_name = None

    def get_next_path(self, next_subgoal, constraint_fns, stage_idx):
        path_constraints = constraint_fns[stage_idx]['path']
        if hasattr(self, "path_solver"):
            path, debug_dict = self.path_solver.solve(
                                        self.environment,
                                        np.concatenate(self.environment.robot.get_current_pose()),
                                        next_subgoal,
                                        self.environment.part_to_pts_dict,
                                        self.environment.moving_part_names,
                                        path_constraints,
                                        None,
                                        from_scratch=self.first_iter,
                                        occlusion_func=None,
                                        curr_ee_pose=np.concatenate(self.environment.robot.get_current_pose())
                                        )
        else:
            path = [np.concatenate(self.environment.robot.get_current_pose()), next_subgoal]
        path = np.stack([np.concatenate(self.environment.robot.get_current_pose()), next_subgoal])
        processed_path = self.process_path(path)[:, :-1]
        self.visualizer.visualize_path(self.environment, processed_path)
        return processed_path

    def process_path(self, path):
        ## Interpolate the paths into pieces
        if len(path) > 0 and np.all(path[0] == path[-1]):
            processed_path = np.concatenate([path, np.zeros((len(path), 1))], axis=-1)
            return processed_path
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            np.concatenate(self.environment.robot.get_current_pose()).reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)

        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        return ee_action_seq
        

    def get_next_subgoal(self, constraint_fns, stage_idx):
        subgoal_constraints = constraint_fns[stage_idx]['subgoal']
        path_constraints = constraint_fns[stage_idx]['path']
        subgoal_pose, debug_dict = self.subgoal_solver.solve(
                np.concatenate(self.environment.robot.get_current_pose()),
                self.environment,
                self.environment.part_to_pts_dict, 
                self.environment.moving_part_names,
                subgoal_constraints,
                path_constraints,
                self.grasp_state > 0,
                None,
                occlusion_func=None,
                from_scratch=self.first_iter)
            
        print(debug_dict)
        self.visualizer.visualize_subgoal(self.environment, subgoal_pose, self.environment.moving_part_names)
        return subgoal_pose

    def plan_paths(self, cost_funs, stage_idx):
        next_subgoal = self.get_next_subgoal(cost_funs, stage_idx)
        next_path = self.get_next_path(next_subgoal, cost_funs, stage_idx)
        self.action_queue = next_path.tolist()
        return

    def grasp_wrapper(self, ):
        def grasp(name):
            self.grasper.grasp(self.environment, )
            self.grasp_state = 1
            self.grasp_name = self.environment.get_grasp_name()
        return grasp
    
    def grasp_postprocess(self, ):
        self.already_grasped = 1
        self.environment.register_moving_part_names(self.grasp_name)

    def release_wrapper(self, ):
        def release():
            self.environment.release()
            self.already_grasped = 0
        return release
    
    def release_postprocess(self, ):
        current_approach = self.environment.robot.get_current_approach()
        self.environment.move_to_point(self.environment.robot.get_current_pose()[0] - current_approach * 0.05, transition=True)
        self.environment.register_moving_part_names()
        self.grasp_name = None

    def execute(self, ):
        while len(self.action_queue) > 0:
            next_action = self.action_queue.pop(0)
            self.environment.move_to_point(next_action)
        if self.grasp_state == 1:
            self.grasp_postprocess()
        else:
            self.release_postprocess()

    def run(self, ):
        os.makedirs(self.task_dir, exist_ok=True)
        self.geometry_parser = self.perception
        constraint_generator = self.constraint_generator
        color_image, _, _ = self.environment.camera.update_frames()
        constraint_generator.task_decomposition_and_geometric_constraint_generation(color_image, self.config['task'], task_dir=self.config['task_dir'])
        cost_function_text = constraint_generator.cost_fns_generation(self.config['task'], task_dir=self.config['task_dir'])
        self.environment.register_geometries(self.config['task_dir'], cost_function_text, self.geometry_parser)
        self.cost_functions, self.cost_functions_text = load_cost_functions(self.config['task_dir'], self)
        self.environment.register_moving_part_names()

        for stage_idx in range(1, len(self.cost_functions) + 1):
            self.plan_paths(self.cost_functions, stage_idx)
            self.execute()
            self.environment.update_stage(stage_idx + 1)
        if hasattr(self.environment, "data_recorder"):
            self.environment.data_recorder.save()