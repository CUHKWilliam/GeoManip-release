from utils.registry import PIPELINES
import open3d as o3d
import copy
import numpy as np
import os
import subprocess
import cv2
from scipy.spatial.transform import Rotation as R
from .base_grasper import GrasperBase

@PIPELINES.register_module()
class GPDGrasper(GrasperBase):
    def __init__(self, config) -> None:
        self.config = config
    
    def generate_candidates(self, env,):
        name = env.get_grasp_name()
        segm_pts_3d = copy.deepcopy(env.part_to_pts_dict[-1][name])
        pcs_mean = segm_pts_3d.mean(0)
        segm_pts_3d -= pcs_mean
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(segm_pts_3d)
        pcd.colors = o3d.utility.Vector3dVector(np.ones((segm_pts_3d.shape[0], 3)))
        o3d.io.write_point_cloud("tmp.pcd", pcd)
        grasp_cfg_path = os.path.join(self.config['gpd_config_path'])
        grasp_bin_path = "detect_grasps"
        output = subprocess.check_output(['{}'.format(grasp_bin_path), '{}'.format(grasp_cfg_path), "tmp.pcd"])
        app_strs = str(output).split("Approach:")[1:]
        approaches = []
        for app_str in app_strs:
            app_str = app_str.strip().split('\\')[0].strip()
            app_vec =  app_str.split(" ")
            app_vec = np.array([float(app_vec[0]), float(app_vec[1]), float(app_vec[2])])
            app_vec = app_vec / np.linalg.norm(app_vec)
            approaches.append(app_vec)
        approaches = np.stack(approaches, axis=0)
        pos_strs = str(output).split("Position:")[1:]
        positions = []
        for pos_str in pos_strs:
            pos_str = pos_str.strip().split('\\')[0].strip()
            pos_vec =  pos_str.split(" ")
            pos_vec = np.array([float(pos_vec[0]), float(pos_vec[1]), float(pos_vec[2])])
            positions.append(pos_vec)
        positions = np.stack(positions, axis=0)

        binormal_strs = str(output).split("Binormal:")[1:]
        binormals = []
        for binormal_str in binormal_strs:
            binormal_str = binormal_str.strip().split('\\')[0].strip()
            binormal_vec =  binormal_str.split(" ")
            binormal_vec = np.array([float(binormal_vec[0]), float(binormal_vec[1]), float(binormal_vec[2])])
            binormal_vec = binormal_vec / np.linalg.norm(binormal_vec)
            binormals.append(binormal_vec)
        binormals = np.stack(binormals, axis=0)
        starts = positions + pcs_mean
        target_quats = []

        for i in range(len(approaches)):
            approach = approaches[i]
            binormal = binormals[i]
            target_quat = env.calculate_quat_from_apporach_and_binormal(approach, binormal)
            target_quats.append(target_quat)
        target_quats = np.stack(target_quats, axis=0)
        target_positions = starts
        subgoal_poses = np.concatenate([target_positions, target_quats], axis=-1)
        return subgoal_poses
    
    def select_grasp(self, subgoal_poses, env):
        costs = []
        
        for subgoal_pose in subgoal_poses:
            ## TODO: implement this later
            # subgoal_pose_homo = T.pose2mat([subgoal_pose[:3], subgoal_pose[3:]])
            # transformed_part_to_pts_dict_3d = transform_geometry(subgoal_pose_homo, part_to_pts_dict_3d_centered, moving_part_names)
            # env.part_to_pts_dict_simulation = copy.deepcopy(transformed_part_to_pts_dict_3d)
            # cost = 0.
            # for constraint in constraints:
            #     cost += constraint()
            # collision_cost = 0.
            # env.part_to_pts_dict_simulation = None

            if self.config['grasp_selection_criterion'] == "direction preference":
                ## TODO: direction preference
                approach = R.from_quat(subgoal_pose[3:]).as_matrix() @ env.robot.approach0
                preference = np.array([0, 0, -1])
                cost = 1 - np.dot(approach, preference)
            else:
                raise NotImplementedError
        
            costs.append(cost)
        costs = np.stack(costs, axis=0)
        subgoal_pose = subgoal_poses[np.argmin(costs)]
        return subgoal_pose
    
    def grasp(self, env, ):
        subgoal_poses = self.generate_candidates(env)
        subgoal_pose = self.select_grasp(subgoal_poses, env)
        subgoal_approach = R.from_quat(subgoal_pose[3:]).as_matrix() @ env.robot.approach0
        pre_subgoal_pose = subgoal_pose.copy()
        pre_subgoal_pose[:3] = subgoal_pose[:3] - subgoal_approach * self.config['pregrasp_approach_offset']
        env.move_to_point(pre_subgoal_pose)
        subgoal_pose[:3] -=  subgoal_approach * self.config['grasp_approach_offset']
        env.move_to_point(subgoal_pose)
        env.grasp()
        return 


