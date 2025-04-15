from ..base_env import EnvBase
import re
import os
import pickle
import numpy as np
from utils.registry import ENVIRONMENT
import cv2
from ..utils import downsample_point_cloud, filter_point_cloud
from scipy.spatial.transform import Rotation as R


@ENVIRONMENT.register_module()
class RealWorldEnv(EnvBase):
    def __init__(self, config):
        super().__init__()
        self.moving_part_names = []
        self.config = config
        self.use_cache = self.config['use_cache']
        self.part_to_pts_dict = []
        self.stage_num = 1

    def register_moving_part_names(self, moving_names=None):
        parts = self.part_to_pts_dict[-1].keys()
        updated_moving_parts = []
        for part in parts:
            if "robot" in part or "gripper" in part :
                updated_moving_parts.append(part)
        if moving_names is not None:
            for moving_name in moving_names:
                updated_moving_parts.append(moving_name)
        self.moving_part_names = updated_moving_parts

    def register_geometries(self, task_dir, cost_function_text, geometry_parser):
        cached_path = f"{task_dir}/part_to_pts_dict.pkl"
        if not self.use_cache or not os.path.exists(cached_path):
            lines = cost_function_text.split("\n")
            geometry_names = []
            grasp_names = {}
            a_part_to_pts_dict  = {}
            ## TODO: use regular expression here
            for line in lines:
                if "get_point_cloud(\"" in line:
                    geometry_name = line.split("get_point_cloud(\"")[1].split("\"")[0]
                    geometry_names.append(geometry_name)
                elif "grasp(\"" in line:
                    geometry_name = line.split("grasp(\"")[1].split("\")")[0]
                    geometry_names.append(geometry_name)
                    grasp_names[stage] = geometry_name
                elif line.strip().startswith("def "):
                    stage = int(line.split("stage_")[1].split("_")[0])
            geometry_names = list(set(geometry_names))
            for geometry_name in geometry_names:
                if "gripper" in geometry_name or "robot" in geometry_name:
                    if "approach" in geometry_name:
                        a_part_to_pts_dict[geometry_name] = np.linspace(self.robot.get_current_pose()[0], self.robot.get_current_pose()[0] +  self.robot.get_current_approach(), 5)
                    elif "binormal" in geometry_name:
                        a_part_to_pts_dict[geometry_name] = np.linspace(self.robot.get_current_pose()[0], self.robot.get_current_pose()[0] + self.robot.get_current_binormal(), 5)
                    else:
                        a_part_to_pts_dict[geometry_name] = np.array(self.robot.get_current_pose()[0])
                else:
                    mask = geometry_parser.try_parse(f"{task_dir}/query_img.png", geometry_name)
                    _, _, pts = self.camera.update_frames(mask)
                    a_part_to_pts_dict[geometry_name] = pts
            self.part_to_pts_dict.append(a_part_to_pts_dict)
            self.grasp_names = grasp_names
            with open(cached_path, 'wb') as f:
                pickle.dump((self.part_to_pts_dict, self.grasp_names), f)
        else:
            with open(cached_path, "rb") as f:
                self.part_to_pts_dict, self.grasp_names = pickle.load(f)
        for part in self.part_to_pts_dict[-1].keys():
            pcs = self.part_to_pts_dict[-1][part]
            if "robot" not in part and "gripper" not in part:
                pcs = downsample_point_cloud(pcs)
                try_filtered_pcs = filter_point_cloud(pcs)
                if len(try_filtered_pcs) > 2:
                    pcs = try_filtered_pcs
            self.part_to_pts_dict[-1][part] = pcs
    
    def get_grasp_name(self, ):
        assert self.stage_num in self.grasp_names.keys()
        return self.grasp_names[self.stage_num]

    def update_stage(self, stage_num):
        self.stage_num = stage_num

    def get_point_cloud_with_timestamp_wrapper(self):
        def get_point_cloud_with_timestamp(name, ts):
            return self.part_to_pts_dict[ts][name]
        return get_point_cloud_with_timestamp

    def calculate_quat_from_apporach_and_binormal(self, approach, binormal):
        assert len(approach) == 3 and len(binormal) == 3
        source_points = np.stack([self.robot.approach0, self.robot.binormal0, np.array([0,0,0])], axis=0)
        target_points = np.stack([approach, binormal, np.array([0, 0, 0])], axis=0)
        transform_mat =  cv2.estimateAffine3D(source_points, target_points, force_rotation=True)[0][:3, :3]
        target_quat = R.from_matrix(transform_mat).as_quat()
        return target_quat