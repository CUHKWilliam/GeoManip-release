import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os


class CameraBase():
    def __init__(self, extrinsic=None):
        ## TODO: camera extrinsic matrix 
        self.extrinsic = extrinsic
        assert self.extrinsic is not None


    def get_cam_rgb(self) -> np.ndarray:
        """Retrieves the current camera frame in RGB format.
        
        Returns:
            np.ndarray: An RGB image array with shape (height, width, 3), 
            where the last dimension contains the red, green, and blue channels 
            in that order. The dtype is expected to be uint8 (0-255 range).
            
        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        """
        raise NotImplementedError
        
    
    def update_frames(self):
        """
        Updates the frames from the camera.
        """
        raise NotImplementedError
        return color_image, image_points, point_cloud
    