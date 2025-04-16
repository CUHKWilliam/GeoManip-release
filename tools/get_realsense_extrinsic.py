import sys
import os
sys.path.append(os.getcwd())
from equipments.robots.ur5 import UR5Robot
from equipments.cameras.realsense.realsense import RealSense
import cv2
import numpy as np
import argparse
import os
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
from matplotlib import pyplot as plt
import csv
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="./tmp")
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

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
    ur5 = UR5Robot(
        {
            "UR_control_ID": "192.168.1.10",
            "UR_receive_ID": "192.168.1.10",
            "UR_robot_home_pose": [0, 0.5, 0.4, 3.1415, 0, 0],
            "approach0": [0, 0, 1],
            "binormal0": [0, 1, 0],
            "eef_to_grasp_dist": 0.15,
        }
    )   
    camera = RealSense(
            {
                "resolution": [640, 480],
                "extrinsic": [[-0.18746593, 0.51792903,-0.83462929, 0.38557143], [ 0.98110535, 0.14011352,-0.13341847, 0.49040391], [0.04784155,-0.84387068,-0.53440945, 0.38047709], [0, 0, 0, 1]]
            }
    )
    input = cv2.waitKey(5)
    cnt = 0
    data_robot =[]
   
    try:
        while input != 'q' and cnt < 30:  # 循环20次
            color_image, points, point_cloud = camera.update_frames()
            # 显示 RGB 图像
            cv2.imshow('RealSense', color_image)
            if cv2.waitKey(1) & 0xFF == 32:
                cnt = cnt + 1
                Image.fromarray(color_image).save(os.path.join(args.save_dir, "image_{:02}.png".format(cnt)))
                robot_pose = np.concatenate(ur5.get_current_pose())
                data_robot.append(robot_pose)
    finally:
        save_robot_data(data_robot, args.save_dir)
        camera.stop()
        cv2.destroyAllWindows()

    # 找棋盘格角点
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.001) # 阈值
    #棋盘格模板规格
    w = 11   # 12 - 1
    h = 8   # 9  - 1
    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    # print(objp)
    # exit()
    objp = objp*0.007  # mm
    # print(objp)
    # exit()e()
    objp_list = []



    # realsense d435 1280
    cameraMatrix = np.array([[608.478 , 0., 331.113], 
                            [0., 608.359, 245.694],
                            [0., 0., 1.]])

    distCoeffs = np.zeros((1, 5))

    imgpoints_2 = [] # 在图像平面的二维点
    objpoints = []
    R_all_chess_to_cam, T_all_chess_to_cam = [], []

    #加载pic文件夹下所有的jpg图像
    fnames = sorted(glob.glob(f'{args.save_dir}/*.png')) 
    print(len(fnames))
    i=0
    valid = []
    for fname in fnames:
        img = cv2.imread(fname)
        # img = cv2.undistort(img, camera_other, distCoeffs_other)
        # 获取画面中心点
        #获取图像的长宽
        h1, w1 = img.shape[0], img.shape[1]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        u, v = img.shape[:2]
        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
        # 如果找到足够点对，将其存储起来
        if ret == True:
            print(fname)
            print("i: {}, corner_num: {}".format(i, len(corners)))
            i = i+1
            # 在原角点的基础上寻找亚像素角点
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            #追加进入世界三维点和平面二维点中
            imgpoints_2.append(corners.reshape(-1,2))
            objpoints.append(objp)

            success, rotation_vector, translation_vector = cv2.solvePnP(np.array([objp]), np.array([corners.reshape(-1,2)]), cameraMatrix, distCoeffs)
            rot_matrix = Rot.from_rotvec(rotation_vector.flatten()).as_matrix()
            chess_to_camera_mat =  np.eye(4)
            chess_to_camera_mat[:3, :3] = rot_matrix 
            chess_to_camera_mat[:3, 3:] = translation_vector
            camera_to_chess_mat = np.linalg.inv(chess_to_camera_mat)
            np.set_printoptions(suppress = True)

            R_all_chess_to_cam.append(chess_to_camera_mat[:3,:3])
            T_all_chess_to_cam.append(chess_to_camera_mat[:3, 3].reshape((3,1)))
            valid.append(True)
        else:
            valid.append(False)
    valid = np.array(valid)
    end_to_base_quat = []
    with open(os.path.join(args.save_dir, "JointStateSteps.csv"), 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if 'marker' in row or "eef" in row:
                continue
            row_list = [float(x) for x in row]
            end_to_base_quat.append(row_list)       
        R_all_base_to_end, T_all_base_to_end = [], []
        end_to_base_quat = np.array(end_to_base_quat)
    for i in range(len(end_to_base_quat)):
        if not valid[i]: continue
        rot = Rot.from_quat(end_to_base_quat[i][3:]).as_matrix() #quat scalar-last (x, y, z, w) format
        homo_matrix = np.eye(4)
        homo_matrix[:3, :3] = rot
        homo_matrix[:3, 3] = end_to_base_quat[i][0:3]
        inv_homo_matrix = np.linalg.inv(homo_matrix)
        R_all_base_to_end.append(inv_homo_matrix[:3, :3])
        T_all_base_to_end.append(inv_homo_matrix[:3, 3].reshape((3,1)))
    R, T = cv2.calibrateHandEye(R_all_base_to_end, T_all_base_to_end, R_all_chess_to_cam, T_all_chess_to_cam, method = 1)#手眼标定
    np.set_printoptions(suppress = True)
    print("hand-eye-calibration R: \n", np.array2string(R, separator=","))
    print("hand-eye-calibration T: \n", np.array2string(T, separator=","))

