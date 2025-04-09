from utils.registry import PIPELINES
import copy
import open3d as o3d
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

@PIPELINES.register_module()
class NaiveGrasper:
    def __init__(self, config) -> None:
        self.config = config
 
    def grasp(self, env, ):
        name = env.get_grasp_name()
        segm_pts_3d = copy.deepcopy(env.part_to_pts_dict[-1][name])
        source_points = np.stack([env.robot.approach0, env.robot.binormal0, np.array([0,0,0])], axis=0)
        target_points = np.stack([np.array([0, 0, -1]), np.array([0, 1, 0]), np.array([0, 0, 0])], axis=0)
        transform_mat =  cv2.estimateAffine3D(source_points, target_points, force_rotation=True)[0][:3, :3]
        target_quat = R.from_matrix(transform_mat).as_quat()
        subgoal_pose = np.concatenate([segm_pts_3d.mean(0), target_quat])
        subgoal_approach = R.from_quat(subgoal_pose[3:]).as_matrix() @ env.robot.approach0
        pre_subgoal_pose = subgoal_pose.copy()
        pre_subgoal_pose[:3] = subgoal_pose[:3] - subgoal_approach * self.config['pregrasp_approach_offset']
        env.robot.move_to_point(pre_subgoal_pose)
        subgoal_pose[:3] -=  subgoal_approach * self.config['grasp_approach_offset']
        env.robot.move_to_point(subgoal_pose )
        env.robot.grasp()
        return np.inf