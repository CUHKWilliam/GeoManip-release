import pickle
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import copy
import utils.transform_utils as T
import cv2
geometry_parser = None
import os
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import time
from utils import *
import numpy as np

camera = None
robot = None
last_pose = None

part_to_pts_dict = []
part_to_pts_dict_simulation = None
moving_parts = []



def get_point_cloud():
    _, _, pcd = camera.update_frames()
    point_cloud = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # camera_pose, camera_quat = get_camera_pose()
    # Rt = np.zeros((4, 4))
    # Rt[:3, :3] = R.from_quat(camera_quat).as_matrix()
    # Rt[:3, 3] = camera_pose
    Rt = camera.EXTRINSIC
    point_cloud = (Rt @ np.concatenate([point_cloud, np.ones((len(point_cloud), 1))], axis=-1).T).T[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def get_masked_pointcloud(mask):
    pcd = get_point_cloud()
    point_cloud = np.asarray(pcd.points)
    point_cloud = point_cloud.reshape(mask.shape[0], mask.shape[1], 3)
    masked_point_cloud = point_cloud[mask.astype(np.bool_)]
    return masked_point_cloud


def get_camera_pose():
    T_end2base = robot.get_T_end2base()
    camera_pose, camera_quat = get_camera_to_base(T_end2base)
    return camera_pose, camera_quat




def parse_geometry(task_dir, obj_geometry_name):
    mask = geometry_parser.try_parse(img=cv2.imread(f"{task_dir}/query_img.png"), geometry_description=obj_geometry_name)
    pts = get_masked_pointcloud(mask)
    return (obj_geometry_name, pts)


def update_part_to_pts_dict():
    a_part_to_pts_dict = copy.deepcopy(part_to_pts_dict[-1])
    assert last_pose is not None
    a_part_to_pts_dict_centered = center_geometry(last_pose, a_part_to_pts_dict, moving_parts)
    a_part_to_pts_dict_updated = transform_geometry(T.pose2mat([np.array(robot.get_current_pose()[0]), robot.get_current_pose()[1]]), a_part_to_pts_dict_centered, moving_parts)
    part_to_pts_dict.append(a_part_to_pts_dict_updated)


rgbs_for_tracking = []
registered_keypoints = None
def register_keypoints(keypoints):
    global registered_keypoints
    registered_keypoints = keypoints