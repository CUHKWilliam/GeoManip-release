import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os
from ..camera_base import CameraBase

class Realsense(CameraBase):
    def __init__(self, config):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.width, self.height = config['resolution']
        self.extrinsic = np.array(config['extrinsic'])

        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)  
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)   
        self.align = rs.align(rs.stream.color)
        self.start()
        ## TODO:
        self.EXTRINSIC = np.array([[-0.18746593, 0.51792903,-0.83462929, 0.38557143],
                [ 0.98110535, 0.14011352,-0.13341847, 0.49040391],
                [ 0.04784155,-0.84387068,-0.53440945, 0.38047709]])
        
    def start(self):
        self.profile = self.pipeline.start(self.config)
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
    
    def update_frames(self):
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
        image_points = np.reshape(vertices, (h, w, 3))


        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)
        vertices = np.asanyarray(points.get_vertices(dims=2)) 
        w = depth_frame.get_width()
        image_Points = np.reshape(vertices , (-1,w,3))

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(image_Points.reshape(-1,3))
        point_cloud.colors = o3d.utility.Vector3dVector(rgb.reshape(-1,3) / 255.0)
        return color_image, image_points, point_cloud
    
    
    def update_image_depth(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

if __name__ == "__main__":
    camera = Realsense()
    color, depth = camera.update_image_depth()