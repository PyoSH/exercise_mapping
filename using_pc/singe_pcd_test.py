import numpy as np
import open3d as o3d
import os
from utils_mapping import *


ROOT_path = '/home/pyo/b1_perception/exercise_mapping'
# dataset_path = os.path.join(ROOT_path, 'data_1', 'pcds_front', 'output_0.pcd')
# dataset_path = os.path.join(ROOT_path, 'data_1', 'pcds_right', 'output_0.pcd')
dataset_path = os.path.join(ROOT_path, 'data_1', 'pcds_left', 'output_280.pcd')
# dataset_path = os.path.join(ROOT_path, 'data_0', 'pcds', 'output_0.pcd')

pcd = npy_from_pcd(dataset_path)


T_s2w_front = np.eye(4)
T_s2w_front[:3, :3] = get_rotation_matrix(roll=np.deg2rad(-90), pitch=np.deg2rad(0), yaw=np.deg2rad(-90))
# T_s2o_front[:3, :3] = get_rotation_matrix(roll=np.deg2rad(90), pitch=np.deg2rad(-90), yaw=np.deg2rad(0)).T
T_s2w_front[0, 3] = 0
T_s2w_front[1, 3] = 0
T_s2w_front[2, 3] = 0

## 해결 요함
T_s2s_left = np.eye(4)
T_s2s_left[:3, :3] = get_rotation_matrix(roll=np.deg2rad(-20), pitch=np.deg2rad(-90), yaw=np.deg2rad(0))
T_s2s_left[0, 3] = 0
T_s2s_left[1, 3] = 0
T_s2s_left[2, 3] = 0
# print(T_s2o_left)
T_s2w_left = np.dot(T_s2w_front, T_s2s_left)

## 해결 요함
T_s2s_right = np.eye(4)
T_s2s_right[:3, :3] = get_rotation_matrix(roll=np.deg2rad(-20), pitch=np.deg2rad(90), yaw=np.deg2rad(0))
T_s2s_right[0, 3] = 0
T_s2s_right[1, 3] = 0
T_s2s_right[2, 3] = 0
# print(T_s2o_right)
T_s2w_right = np.dot(T_s2w_front, T_s2s_right)

applied_pcd = apply_transform2cloud(pcd, T_s2w_left)

# save_pcd_from_npy(applied_pcd, os.path.join(ROOT_path, 'data_0', '0_app.pcd'))
save_pcd_from_npy(applied_pcd, os.path.join(ROOT_path, 'data_1', '280_left.pcd'))