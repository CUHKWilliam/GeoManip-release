import open3d as o3d
import numpy as np

def downsample_point_cloud(point_clouds, voxel_size=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downpcd.points)

def filter_point_cloud(point_clouds, nb_points=16, radius=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds)
    pcd, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return np.asarray(pcd.points)