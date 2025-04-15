from utils.registry import PIPELINES
import copy
import open3d as o3d
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from .base_grasper import GrasperBase

@PIPELINES.register_module()
class NaiveGrasper(GrasperBase):
    def __init__(self, config) -> None:
        self.config = config
    
    def grasp(self, env, ):
        name = env.get_grasp_name()
        segm_pts_3d = copy.deepcopy(env.part_to_pts_dict[-1][name])
        target_approach, target_binormal = np.array([0, 0, -1]), np.array([0, 1, 0])
        target_quat = env.calculate_quat_from_apporach_and_binormal(target_approach, target_binormal)
        subgoal_pose = np.concatenate([segm_pts_3d.mean(0), target_quat])
        subgoal_approach = R.from_quat(subgoal_pose[3:]).as_matrix() @ env.robot.approach0
        pre_subgoal_pose = subgoal_pose.copy()
        pre_subgoal_pose[:3] = subgoal_pose[:3] - subgoal_approach * self.config['pregrasp_approach_offset']
        env.robot.move_to_point(pre_subgoal_pose)
        subgoal_pose[:3] -=  subgoal_approach * self.config['grasp_approach_offset']
        env.robot.move_to_point(subgoal_pose )
        env.robot.grasp()
        return np.inf