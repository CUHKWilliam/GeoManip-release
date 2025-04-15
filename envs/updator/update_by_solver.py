from utils.registry import ENVIRONMENT
import copy
import numpy as np
import  utils.transform_utils as T

@ENVIRONMENT.register_module()
class UpdateBySolver:
    def __init__(self, config):
        self.config = config
    
    def center_geometry(self, ee_pose, a_part_to_pts_dict_, moving_part_names,):
        a_part_to_pts_dict = copy.deepcopy(a_part_to_pts_dict_)
        if len(ee_pose[3:]) == 3:
            ee_pose = np.concatenate([ee_pose[:3], T.euler2quat(ee_pose[3:])])
        ee_pose_homo = T.pose2mat([ee_pose[:3], ee_pose[3:]])
        centering_transform = np.linalg.inv(ee_pose_homo)
        for key in a_part_to_pts_dict.keys():
            if key in moving_part_names:
                a_part_to_pts_dict[key] = np.dot(a_part_to_pts_dict[key], centering_transform[:3, :3].T) + centering_transform[:3, 3]
            else:
                a_part_to_pts_dict[key] = a_part_to_pts_dict[key]
        return a_part_to_pts_dict
    def transform_geometry(self, transform, a_part_to_pts_dict_, moving_part_names,):
        a_part_to_pts_dict = copy.deepcopy(a_part_to_pts_dict_)
        for part_name in a_part_to_pts_dict.keys():
            if part_name in moving_part_names:
                part_pts = a_part_to_pts_dict[part_name]
                a_part_to_pts_dict[part_name] = part_pts + transform[:3, 3]
        return a_part_to_pts_dict

    def update(self, env):
        a_part_to_pts_dict = copy.deepcopy(env.part_to_pts_dict[-1])
        assert env.last_pose is not None
        a_part_to_pts_dict_centered = self.center_geometry(env.last_pose, a_part_to_pts_dict, env.moving_part_names)
        a_part_to_pts_dict_updated = self.transform_geometry(T.pose2mat([np.array(env.robot.get_current_pose()[0]), env.robot.get_current_pose()[1]]), a_part_to_pts_dict_centered, env.moving_part_names)
        env.part_to_pts_dict.append(a_part_to_pts_dict_updated)
