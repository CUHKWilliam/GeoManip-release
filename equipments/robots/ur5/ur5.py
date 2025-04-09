from utils.registry import ROBOTS
from ..robot_base import RobotBase
#=====================#
# function: ur rtde and other ur related robotics functions
#=====================#

import sys

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.geometry_utils import get_transformation_matrix, rotation_vector_to_quaternion
from .gripper import Gripper
import pickle

@ROBOTS.register_module()
class URRobot(RobotBase):
    def __init__(self, config, ):
        self.UR_control_ID = config["UR_control_ID"]
        self.UR_receive_ID = config['UR_receive_ID']
        self.approach0, self.binormal0 = config['UR_approach0'], config['UR_binormal0']
        self.eef_to_grasp_dist = config['eefto_grasp_dist']
        self.rtde_init()
        self.soft_gripper = Gripper()
        self.dz = 0.12
        ## TODO: home position
        self.move_to_point(np.array(config['UR_robot_home_pose']))
        self.gripper_state = 0

    def rtde_init(self):
        self.rtde_c = RTDEControl(self.UR_control_ID)
        self.rtde_r = RTDEReceive(self.UR_receive_ID)

    #==== get ur data from rtde
    def get_jq(self):
        return self.rtde_r.getActualQ()
    
    def get_end_pose(self):
        # end pose in base frame bTe
        return self.rtde_r.getActualTCPPose()

    def get_gripper_state(self):
        return self.gripper_state
    
   #==== set ur cmd through rtde
    def move_to_point(self, tar, v=0.1, a=0.1):
        # moveL in the base frame
        self.rtde_c.moveL(tar, v, a)

    def grasp(self,):
        self.soft_gripper.gripper_close()
        self.gripper_state = 1

    def release(self, ):
        self.soft_gripper.gripper_open()
        self.gripper_state = 0

    def get_current_pose(self,):
        # Stop the RTDE control script
        current_pose = self.get_end_pose()
        current_pos = current_pose[:3]
        current_quat = rotation_vector_to_quaternion(current_pose[3:])
        return current_pos, current_quat