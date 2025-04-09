import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import *
from gripper_util import Gripper

class RobotBase():
    def __init__(self, config):
        raise NotImplementedError
    
    def move_to_point(self, target: np.ndarray) -> None:
        """Move the gripper to the specified target pose.
        
        Args:
            target: A numpy array of shape (7,) containing [x, y, z, rx, ry, rz, rw] where:
                - x, y, z: Cartesian coordinates (in meters) for the gripper position
                - rx, ry, rz, rw: Quaternion components for the gripper orientation
            
        Raises:
            NotImplementedError: If the method is not implemented by the subclass
            ValueError: If target array is not of shape (7,)
        """
        raise NotImplementedError

    def grasp(self,) -> None:
        '''Close the gripper
        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        '''
        raise NotImplementedError

    def release(self, ) -> None:
        """ Release the gripper
        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        """
        raise NotImplementedError
        

    def get_current_pose(self,) -> tuple[np.ndarray, np.ndarray]:
        """Retrieves the current camera frame in RGB format.
    
        Returns:
            np.ndarray: A (x, y, z) gripper position array with shape (3,),
            np.ndarray: A (x, y, z, w) gripper quaternion array with shape (4,)

        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        """
        raise NotImplementedError
        