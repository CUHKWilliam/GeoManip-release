import numpy as np
import open3d as o3d
import os
from ..pipeline_base import PipelineBase
from utils.registry import *
import utils.builder as builder
from .utils import *
from scipy.spatial.transform import Rotation as R
from utils.registry import PIPELINES

@PIPELINES.register_module()
class GeomanipPipeline(PipelineBase):
    def __init__(self, config):
        self.first_iter = True
        self.action_queue = None
        self.grasp_state = 0
        self.grasp_part_name = None
        self.already_grasped = 0
        self.config = config

    def get_next_path(self, next_subgoal, constraint_fns, stage_idx):
        path_constraints = constraint_fns[stage_idx]['path']
        path, debug_dict = path_solver.solve(
                                    np.concatenate(self.env.robot.get_current_pose()),
                                    self.env,
                                    next_subgoal,
                                    self.env.part_to_pts_dict,
                                    self.env.moving_parts,
                                    path_constraints,
                                    None,
                                    from_scratch=self.first_iter,
                                    occlusion_func=None,
                                    curr_ee_pose=np.concatenate(self.env.robot.get_current_pose())
                                    )
        path = np.stack([np.concatenate(self.env.robot.get_current_pose()), next_subgoal])
        processed_path = process_path(path)
        self.visualizer.visualize_path(processed_path)
        return processed_path

    def process_path(self, path):
        ## Interpolate the paths into pieces
        if len(path) > 0 and np.all(path[0] == path[-1]):
            processed_path = np.concatenate([path, np.zeros((len(path), 1))], axis=-1)
            return processed_path
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            np.concatenate(self.env.robot.get_current_pose()).reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    config['interpolate_pos_step_size'],
                                                    config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)

        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        return ee_action_seq
        

    def get_next_subgoal(self, constraint_fns, stage_idx):
        global grasp_state
        subgoal_constraints = self.constraint_fns[stage_idx]['subgoal']
        path_constraints = self.constraint_fns[stage_idx]['path']
        result = subgoal_constraints[0]()
        if isinstance(result, dict):
            subgoal_poses = result['subgoal_poses']
            grasp_constraints = subgoal_constraints[1:]
            global grasp_part_name
            grasp_part_name = cost_functions_text[1]['subgoal'].split("grasp(\"")[1].split("\")")[0]
            subgoal_pose = select_grasp_with_constraints(subgoal_poses, grasp_constraints, grasp_part_name)
            grasp_state = 1
        elif result is None:
            grasp_state = 0
            self.env.register_moving_part_names()
            return None
        else:
            subgoal_pose, debug_dict = subgoal_solver.solve(
                    np.concatenate(self.env.robot.get_current_pose()),
                    self.env,
                    self.env.part_to_pts_dict, 
                    self.env.moving_parts,
                    subgoal_constraints,
                    path_constraints,
                    grasp_state > 0,
                    None,
                    occlusion_func=None,
                    from_scratch=self.first_iter)
            print(debug_dict)
        self.visualizer.visualize_subgoal(subgoal_pose, self.env.moving_parts)
        return subgoal_pose

    def plan_paths(self, cost_funs, stage_idx):
        global action_queue
        next_subgoal = self.get_next_subgoal(cost_funs, stage_idx)
        next_path = self.get_next_path(next_subgoal, cost_funs, stage_idx)
        action_queue = self.next_path.tolist()
        return

    def grasp_postprocess(self, ):
        global already_grasped
        if already_grasped == 1:
            return
        current_approach = R.from_quat(self.env.robot.get_current_pose()[1]).as_matrix() @ self.robot.approach0
        self.env.robot.move_to_point(np.concatenate([self.env.robot.get_current_pose()[0] + current_approach * 0.2, euler_to_rotation_vector(R.from_quat(self.env.robot.get_current_pose()[1]).as_euler("XYZ"))], axis=0))
        self.env.robot.grasp()
        obj_name = grasp_part_name.split("of")[-1].strip()
        moving_part_names = []
        for name in list(self.env.part_to_pts_dict[-1].keys()):
            if obj_name in name:
                moving_part_names.append(name)
        self.env.register_moving_part_names(moving_part_names)
        already_grasped = 1

    def release_postprocess(self, ):
        self.env.robot.release()
        global already_grasped
        already_grasped = 0

    def execute(self, ):
        self.env.last_pose = np.concatenate(self.env.robot.get_current_pose())
        while len(action_queue) > 0:
            next_action = action_queue.pop(0)
            self.env.robot.move_to_point(next_action)
        self.env.update_part_to_pts_dict()
        if self.grasp_state == 1:
            self.grasp_postprocess()
        else:
            self.release_postprocess()

    def run(self, ):
        os.makedirs(self.task_dir, exist_ok=True)
        self.env.geometry_parser = GeometryParser(task_dir=task_dir)
        constraint_generator = GeometricAndCodeGenerator(config['constraint_generator']['config'])
        color_image, depth_image = camera.update_image_depth()
        constraint_generator.task_decomposition_and_geometric_constraint_generation(color_image, task, task_dir=task_dir)
        cost_function_text = constraint_generator.cost_fns_generation(task_dir=task_dir)
        self.env.register_geometries(task_dir, cost_function_text)
        cost_functions, cost_functions_text = load_cost_functions(task_dir)
        self.env.register_moving_part_names()

        for stage_idx in range(len(cost_functions)):
            plan_paths(cost_functions, stage_idx + 1)
            execute()
