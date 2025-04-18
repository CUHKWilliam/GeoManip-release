import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os
from ..camera_base import CameraBase
from utils.registry import CAMERAS
from PIL import Image
DEBUG = True


@CAMERAS.register_module()
class RealSense(CameraBase):
    def __init__(self, config):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.width, self.height = config['resolution']
        self.extrinsic = np.array(config['extrinsic'])

        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)  
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)   
        self.align = rs.align(rs.stream.color)
        if not DEBUG:
            self.start()
        
    def start(self):
        self.profile = self.pipeline.start(self.config)
        sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        # Set the exposure anytime during the operation
        sensor.set_option(rs.option.exposure, 200.000)
        return None

    def stop(self):
        self.pipeline.stop()
        return None
    
    def get_color_intrinsics(self):
        color_intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
        return color_intrinsics
        

    def get_cam_rgb(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        rgb = color_image[...,::-1]
        return rgb
    
    def update_frames(self, mask=None):
        if DEBUG:
            img = np.asarray(Image.open("./realsense_debug_scene.png").convert("RGB"))
            depth = np.zeros((img.shape[0], img.shape[1])).astype(np.float32)
            points = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.float32)
            if mask is None:
                return img, depth, points.reshape(((-1, 3)))
            else:
                return img[mask], depth[mask], points[mask].reshape(((-1, 3)))
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = color_image[:, :, ::-1]
        rgb = color_image
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)
        vertices = np.asanyarray(points.get_vertices(dims=2))  
        w = depth_frame.get_width()
        h = depth_frame.get_height()

        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)
        vertices = np.asanyarray(points.get_vertices(dims=2)) 
        w = depth_frame.get_width()
        image_points = np.reshape(vertices , (-1,w,3))
        points = image_points.reshape(-1, 3)
        points = (self.extrinsic @ np.concatenate([points, np.ones((len(points), 1))], axis=-1).T).T[:, :3]
        points = points.reshape((-1, w, 3))

        if mask is None:
            return color_image, depth_image, points.reshape(-1,3)
        else:
            return color_image[mask], depth_image[mask], points[mask].reshape(-1,3)
    
    
    def update_image_depth(self):
        if DEBUG:
            return np.zeros((128, 128, 3)).astype(np.uint8), None
        
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())[:, :, ::-1]
        return color_image, depth_image

if __name__ == "__main__":
    camera = RealSense()
    color, depth = camera.update_image_depth()
    cv2.imwrite("obs.png", color)