from ..base_env import EnvBase
import re
import os
import pickle
import numpy as np
from utils.registry import ENVIRONMENT

@ENVIRONMENT.register_module()
class RealWorldEnv(EnvBase):
    def __init__(self, config):
        super().__init__()
        self.moving_part_names = []
        self.config = config
        self.use_cache = self.config['use_cache']

    def register_moving_part_names(self, moving_names=None):
        moving_name_config = self.config['moving_names']
        parts = part_to_pts_dict[-1].keys()
        updated_moving_parts = []
        for part in parts:
            if "robot" in part or "gripper" in part :
                updated_moving_parts.append(part)
        if moving_names is not None:
            for moving_name in moving_names:
                if not re.match(moving_name_config['blacklist'], moving_name):
                    updated_moving_parts.append(moving_name)
        self.moving_part_names = updated_moving_parts

    def register_geometries(self, task_dir, cost_function_text, ):
        cached_path = f"{task_dir}/part_to_pts_dict.pkl"
        global part_to_pts_dict
        if not self.use_cache or not os.path.exists(cached_path):
            lines = cost_function_text.split("\n")
            geometry_names = []
            a_part_to_pts_dict  = {}
            ## TODO: use regular expression here
            for line in lines:
                if "get_point_cloud(\"" in line:
                    geometry_name = line.split("get_point_cloud(\"")[1].split("\"")[0]
                    geometry_names.append(geometry_name)
                elif "grasp(\"" in line:
                    geometry_name = line.split("grasp(\"")[1].split("\")")[0]
                    geometry_names.append(geometry_name)
            geometry_names = list(set(geometry_names))
            for geometry_name in geometry_names:
                if "gripper" in geometry_name or "robot" in geometry_name:
                    if "approach" in geometry_name:
                        a_part_to_pts_dict[geometry_name] = np.linspace(robot.get_current_pose()[0], robot.get_current_pose()[0] + R.from_quat(robot.get_current_pose()[1]).as_matrix() @ APPROACH0 , 5)
                    elif "binormal" in geometry_name:
                        a_part_to_pts_dict[geometry_name] = np.linspace(robot.get_current_pose()[0], robot.get_current_pose()[0] + R.from_quat(robot.get_current_pose()[1]).as_matrix() @ BINORMAL0, 5)
                    else:
                        a_part_to_pts_dict[geometry_name] = np.array(robot.get_current_pose()[0])
                else:
                    mask = geometry_parser.try_parse(img=cv2.imread(f"{task_dir}/query_img.png"), geometry_description=geometry_name)
                    pts = get_masked_pointcloud(mask)
                    a_part_to_pts_dict[geometry_name] = pts
            part_to_pts_dict.append(a_part_to_pts_dict)
            with open(cached_path, 'wb') as f:
                pickle.dump(part_to_pts_dict, f)
        else:
            with open(cached_path, "rb") as f:
                part_to_pts_dict = pickle.load(f)
        for part in part_to_pts_dict[-1].keys():
            pcs = part_to_pts_dict[-1][part]
            if "robot" not in part and "gripper" not in part:
                pcs = downsample_point_cloud(pcs)
                try_filtered_pcs = filter_point_cloud(pcs)
                if len(try_filtered_pcs) > 0:
                    pcs = try_filtered_pcs
            part_to_pts_dict[-1][part] = pcs