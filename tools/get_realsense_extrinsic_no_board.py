import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from equipments.robots.ur5 import UR5Robot
from equipments.cameras.realsense.realsense import RealSense
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import cv2
import time

camera = RealSense(
            {
                "resolution": [640, 480],
                "extrinsic": [[-0.18746593, 0.51792903,-0.83462929, 0.38557143], [ 0.98110535, 0.14011352,-0.13341847, 0.49040391], [0.04784155,-0.84387068,-0.53440945, 0.38047709], [0, 0, 0, 1]]
            }
    )
ur5 = UR5Robot(
        {
            "UR_control_ID": "192.168.1.10",
            "UR_receive_ID": "192.168.1.10",
            "UR_robot_home_pose": [0, 0.5, 0.4, 3.1415, 0, 0],
            "approach0": [0, 0, 1],
            "binormal0": [0, 1, 0],
            "eef_to_grasp_dist": 0.15,
        }
    )   
import ipdb;ipdb.set_trace()
ur5.grasp()
import ipdb;ipdb.set_trace()

def manual_control():
    while True:
        ch = input("")
        if  ch == "c":
            break
    return ur5.get_current_pose()

def get_mask_by_color(rgb):
    rgb = rgb.astype(np.float32)
    rgb = rgb / 255.
    rgb = rgb / np.linalg.norm(rgb, axis=-1)[:, :, None]
    red_vec = np.array([1., 0., 0.])
    red_vec = red_vec / np.linalg.norm(red_vec)
    dot = (rgb * red_vec).sum(-1)
    mask = dot > 0.9
    mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if mask.sum() == 0:
        print("no mask!")
        return None
    contour, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    mask2 = np.zeros_like(mask).astype(np.uint8)
    biggest_area = -1
    biggest = None
    for con in contour:
        area = cv2.contourArea(con)
        if biggest_area < area:
            biggest_area = area
            biggest = con

    # fill in the contour
    cv2.drawContours(mask2, [biggest], -1, 255, -1)
    mask2 = mask2 > 0
    return mask2

cam_poses = []


## TODO: mode 2:
robot_poses = []
for _ in range(20):
    robot_pos, robot_ori = manual_control()
    robot_poses.append(robot_pos)
    rgb, _, pcd = camera.update_frames()
    pcs = np.asarray(pcd.points)
    cv2.imwrite('debug.png', rgb[:, :, ::-1])
    red_mask = get_mask_by_color(rgb)
    if red_mask is None:
        continue
    cv2.imwrite("debug2.png", (red_mask).astype(np.uint8) * 255)
    pcs = pcs[red_mask.reshape(-1)]
    pcs = pcs[np.logical_not(np.isnan(pcs).any(1))]
    cam_pos = pcs.mean(0)
    cam_poses.append(cam_pos)
robot_poses = np.stack(robot_poses, axis=0)
## end TODO

import ipdb;ipdb.set_trace()
trans_mat = cv2.estimateAffine3D(np.asarray([
        cam_poses
    ]), 
    np.asarray([
       robot_poses
    ]))[1]

print(trans_mat)