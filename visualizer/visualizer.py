import open3d as o3d
import numpy as np
import matplotlib
import cv2
import utils.transform_utils as T
from utils.utils import filter_points_by_bounds, batch_transform_points, draw_arrow
from scipy.spatial.transform import Rotation as R
import os
from utils.registry import VISUALIZERS
from PIL import Image

def add_to_visualize_buffer(visualize_buffer, visualize_points, visualize_colors):
    assert visualize_points.shape[0] == visualize_colors.shape[0], f'got {visualize_points.shape[0]} for points and {visualize_colors.shape[0]} for colors'
    if len(visualize_points) == 0:
        return
    assert visualize_points.shape[1] == 3
    assert visualize_colors.shape[1] == 3
    # assert visualize_colors.max() <= 1.0 and visualize_colors.min() >= 0.0
    visualize_buffer["points"].append(visualize_points)
    visualize_buffer["colors"].append(visualize_colors)

def generate_nearby_points(point, num_points_per_side=5, half_range=0.005):
    if point.ndim == 1:
        offsets = np.linspace(-1, 1, num_points_per_side)
        offsets_meshgrid = np.meshgrid(offsets, offsets, offsets)
        offsets_array = np.stack(offsets_meshgrid, axis=-1).reshape(-1, 3)
        nearby_points = point + offsets_array * half_range
        return nearby_points.reshape(-1, 3)
    else:
        assert point.shape[1] == 3, "point must be (N, 3)"
        assert point.ndim == 2, "point must be (N, 3)"
        # vectorized version
        offsets = np.linspace(-1, 1, num_points_per_side)
        offsets_meshgrid = np.meshgrid(offsets, offsets, offsets)
        offsets_array = np.stack(offsets_meshgrid, axis=-1).reshape(-1, 3)
        nearby_points = point[:, None, :] + offsets_array
        return nearby_points

@VISUALIZERS.register_module()
class SubgoalPathVisualizer:
    def __init__(self, config,):
        self.config = config
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.color = np.array([0.05, 0.55, 0.26])

    def show_img(self, rgb):
        cv2.imshow('img', rgb[..., ::-1])
        cv2.waitKey(0)
        print('showing image, click on the window and press "ESC" to close and continue')
        cv2.destroyAllWindows()
    
    def show_pointcloud(self, env, points, colors, subgoal_pose_homos, save=None, return_pcd=False):
        # transform to viewer frame
        # points = np.dot(points, self.world2viewer[:3, :3].T) + self.world2viewer[:3, 3]
        # clip color to [0, 1]
        colors = np.clip(colors, 0.0, 1.0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        for subgoal_pose_homo in subgoal_pose_homos:
            start = subgoal_pose_homo[:3, -1]
            R, pos = subgoal_pose_homo[:3, :3], subgoal_pose_homo[:3, -1]
            end = np.dot(env.robot.approach0 * 0.3, R.T) + pos
            pcd = draw_arrow(pcd, start, end, np.array([1, 0, 0]))
            end = np.dot(env.robot.binormal0 * 0.05, R.T) + pos
            pcd = draw_arrow(pcd, start, end, np.array([0, 1, 0]))
        o3d.io.write_point_cloud(save, pcd)
        if return_pcd:
            return np.asarray(pcd.points), np.asarray(pcd.colors)

    def _get_scene_points_and_colors(self, env):
        # scene
        scene_points = []
        scene_colors = []
        rgb, _, pcs = env.camera.update_frames()
        return pcs, rgb.reshape(-1, 3)

    def visualize_subgoal(self, env, subgoal_pose, moving_part_names):
        if subgoal_pose is None or len(subgoal_pose) == 0:
            return
        visualize_buffer = {
            "points": [],
            "colors": []
        }
        # scene
        rgb, _, pcs = env.camera.update_frames()
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        self.show_pointcloud(env, pcs, rgb.reshape(-1, 3), subgoal_pose_homo[None, ...], self.config['ply_save_path'])

    def visualize_path(self, env, path, save_path="debug.ply", return_pcd=False):
        if path is None:
            return
        if isinstance(path, list):
            path = np.array(path)
        visualize_buffer = {
            "points": [],
            "colors": []
        }
        # scene
        scene_points, scene_colors = self._get_scene_points_and_colors(env, )
        add_to_visualize_buffer(visualize_buffer, scene_points, scene_colors)
        # draw curve based on poses
        for t in range(len(path) - 1):
            start = path[t][:3]
            end = path[t + 1][:3]
            num_interp_points = int(np.linalg.norm(start - end) / 0.0002)
            interp_points = np.linspace(start, end, num_interp_points)
            interp_colors = np.tile([0.0, 0.0, 0.0], (num_interp_points, 1))
            # add a tint of white (the higher the j, the more white)
            whitening_coef = 0.3 + 0.5 * (t / len(path))
            interp_colors = (1 - whitening_coef) * interp_colors + whitening_coef * np.array([1, 1, 1])
            add_to_visualize_buffer(visualize_buffer, interp_points, interp_colors)
        # subsample path with a fixed step size
        step_size = 0.01
        # subpath = [path[0]]
        # for i in range(1, len(path) - 1):
        #     dist = np.linalg.norm(np.array(path[i][:3]) - np.array(subpath[-1][:3]))
        #     if dist > step_size:
        #         subpath.append(path[i])
        # subpath.append(path[-1])
        subpath = path
        path = np.array(subpath)
        path_length = path.shape[0]
        # path points

        poses_homo = T.convert_pose_quat2mat(path[:, :7])
        return self.show_pointcloud(env, scene_points, scene_colors, poses_homo, self.config['ply_save_path'], return_pcd)
    
@VISUALIZERS.register_module()
class ImageVisualizer:
    def __init__(self, config):
        self.config = config
        save_dir = config['save_dir']
        for file in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, file))

    def show_img(self, rgb, img_name=None, ):
        if self.config['pop_window']:
            cv2.imshow('img', rgb[..., ::-1])
            print(f'showing image of {img_name}, click on the window and press "ESC" to close and continue')
        else:
            save_path = os.path.join(self.config['save_dir'], img_name + ".png")
            Image.fromarray(rgb).save(save_path)
    