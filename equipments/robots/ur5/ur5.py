DEBUG = False

from utils.registry import ROBOTS
from ..robot_base import RobotBase
#=====================#
# function: ur rtde and other ur related robotics functions
#=====================#

import sys
if not DEBUG:
    from rtde_control import RTDEControlInterface as RTDEControl
    from rtde_receive import RTDEReceiveInterface as RTDEReceive
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.utils import rotation_vector_to_quaternion
from .gripper import Gripper
import pickle
from utils.utils import euler_to_rotation_vector, quaternion_to_rotation_vector
from utils.utils import ensure_shortest_quaterion

@ROBOTS.register_module()
class UR5Robot(RobotBase):

    def __init__(self, config, ):
        self.gripper_state = 0
        self.config = config
        self.approach0, self.binormal0 = np.array(config['approach0']), np.array(config['binormal0'])

        if DEBUG:
            return
        self.UR_control_ID = config["UR_control_ID"]
        self.UR_receive_ID = config['UR_receive_ID']
        self.eef_to_grasp_dist = config['eef_to_grasp_dist']
        self.rtde_init()
        self.soft_gripper = Gripper()
        ## TODO: home position
        # self.move_to_point(np.array(config['UR_robot_home_pose']))

    def rtde_init(self):
        self.rtde_c = RTDEControl(self.UR_control_ID)
        self.rtde_r = RTDEReceive(self.UR_receive_ID)

    #==== get ur data from rtde
    def get_jq(self):
        return self.rtde_r.getActualQ()
    
    def get_end_pose(self):
        if DEBUG:
            return np.array([0., 0, 0., 0, 0, 0])
        # end pose in base frame bTe
        return self.rtde_r.getActualTCPPose()

    def get_gripper_state(self):
        return self.gripper_state
    
   #==== set ur cmd through rtde
    def move_to_point(self, tar, v=0.1, a=0.1, transition=False):
        if DEBUG:
            return
        if len(tar) == 6: # input is x, y, z, rx, ry, rz
            tar = np.concatenate([tar[:3], rotation_vector_to_quaternion(tar[3:])])

        if not transition:
            assert len(tar) == 7 # input target is x, y, z + quaternion
        else:
            assert len(tar) == 3

        if not transition:
            tar[3:] = ensure_shortest_quaterion(self.get_current_pose()[1], tar[3:])
            tar_rot = euler_to_rotation_vector(R.from_quat(tar[3:]).as_euler("XYZ"))
            tar = np.concatenate([tar[:3], tar_rot])
            self.rtde_c.moveL(tar, v, a)
        else:
            tar_rot = euler_to_rotation_vector(R.from_quat(self.get_current_pose()[1]).as_euler("XYZ"))
            tar = np.concatenate([tar, tar_rot])
            self.rtde_c.moveL(tar, v, a)

    def grasp(self,):
        if DEBUG:
            return
        self.soft_gripper.gripper_close()
        self.gripper_state = 1

    def release(self, ):
        if DEBUG:
            return
        self.soft_gripper.gripper_open()
        self.gripper_state = 0

    def get_current_pose(self,):
        if DEBUG:
            return np.array([0., 0, 0.]), np.array([0,  0, 0, 1])
        # Stop the RTDE control script
        current_pose = self.get_end_pose()
        current_pos = current_pose[:3]
        current_quat = rotation_vector_to_quaternion(current_pose[3:])
        return current_pos, current_quat
    
    def get_current_approach(self,):
        return R.from_quat(self.get_current_pose()[1]).as_matrix() @ self.config['approach0']
    
    def get_current_binormal(self,):
        return R.from_quat(self.get_current_pose()[1]).as_matrix() @ self.config['binormal0']
