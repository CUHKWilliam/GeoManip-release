from utils.registry import VISUALIZERS
import pickle
import numpy as np
from utils.utils import get_linear_interpolation_steps, spline_interpolate_poses

@VISUALIZERS.register_module()
class DataRecorderV1:
    def __init__(self, config):
        self.config = config
        self.datas = []
        self.is_grasped = 1

    def move_and_log(self, env, target_pose):
        '''
        
        '''
        full_control_points = np.concatenate([
            np.concatenate(env.robot.get_current_pose()).reshape(1, -1),
            [target_pose],
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        for pose in dense_path:
            self.datas.append({
                'rgb': env.camera.update_frames()[0],
                'pose': np.concatenate([pose[:7] - env.robot.get_current_pose(), np.array([self.is_grasped])], axis=0),
            })
            env.move_to_point(pose[:7])

    def grasp_and_log(self, env):
        self.datas.append({
            'rgb': env.camera.update_frames()[0],
            'pose': np.concatenate([np.concatenate(env.robot.get_current_pose()) * 0., np.array([[1]])], axis=0),
        })
        env.robot.grasp()
        self.is_grasped = 1
    
    def release_and_log(self, env):
        self.datas.append({
            'rgb': env.camera.update_frames()[0],
            'pose': np.concatenate([np.concatenate(env.robot.get_current_pose()) * 0., np.array([[0]])], axis=0),
        })
        self.robot.release()
        self.is_grasped = 0

    def save(self, ):
        with open(self.config['save_path'], 'wb') as f:
            pickle.dump(self.datas, f)
        print(f"data saved to {self.config['save_path']}")