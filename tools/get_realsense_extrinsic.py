from equipments.robots.ur5 import UR5Robot
from equipments.cameras.realsense.realsense import RealSense
import cv2
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="./tmp")
args = parser.parse_args()

os.makedirs(args.save_dir)

def save_robot_data(data_robot, save_dir):

    data_robot = np.array([data_robot]).squeeze()

    filename_csv = os.path.join(save_dir, 'JointStateSteps.csv')
    filename_txt = os.path.join(save_dir, 'JointStateSteps.txt')

    # 列标题
    header = 'eef'

    # 保存到CSV文件
    with open(filename_csv, 'w') as file:
        # 写入列标题
        file.write(header + '\n')
        # 使用numpy.savetxt写入数据
        np.savetxt(file, data_robot, delimiter=',', fmt='%f')


    # 保存到CSV文件
    with open(filename_txt, 'w') as file:
        # 写入列标题
        file.write(header + '\n')
        # 使用numpy.savetxt写入数据
        np.savetxt(file, data_robot, delimiter=',', fmt='%f')


if __name__ == "__main__":
    ur5 = UR5Robot(gripper_flag=False)
    camera = RealSense()
    camera.start()
    input = cv2.waitKey(5)
    cnt = 0
    data_robot =[]
   
    try:
        while input != 'q' and cnt < 11:  # 循环20次
            color_image, points, point_cloud = camera.update_frames()
            # 显示 RGB 图像
            cv2.imshow('RealSense', color_image)
            if cv2.waitKey(1) & 0xFF == 32:
                cnt = cnt + 1
                camera.save_image_points(color_image, point_cloud, cnt)
                robot_pose = ur5.get_current_pose()[1]
                data_robot.append(robot_pose)
    finally:
        save_robot_data(data_robot, args.save_dir)
        camera.stop()
        cv2.destroyAllWindows()
