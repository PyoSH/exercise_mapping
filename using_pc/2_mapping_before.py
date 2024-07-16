#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 22:05:01 2024

@author: pyo
"""
import numpy as np

from utils_mapping import *
from data_processing import *

if __name__ =='__main__':
    dataset_path = '/home/pyo/b1_perception/exercise_mapping'
    pcd_front_path = os.path.join(dataset_path, 'pcds_front')
    pcd_left_path = os.path.join(dataset_path, 'pcds_left')
    pcd_right_path = os.path.join(dataset_path, 'pcds_right')
    odom_path = dataset_path

    # ### 1. Load lists
    obj_front = MATCH_PC_ODOM(dataset_path, pcd_front_path, odom_path)
    obj_front.run()

    # obj_left = MATCH_PC_ODOM(dataset_path, pcd_left_path, odom_path)
    # obj_left.run()

    obj_right = MATCH_PC_ODOM(dataset_path, pcd_right_path, odom_path)
    obj_right.run()

    ### transform clouds with (sensor -> odom) coordinate frame T,
    T_s2o_front = np.eye(4)
    T_s2o_front[:3, :3] = get_rotation_matrix(np.deg2rad(90), np.deg2rad(-90), 0).T
    T_s2o_front[0, 3] = 0
    T_s2o_front[1, 3] = 0
    T_s2o_front[2, 3] = 0
    # print(T_s2o_front)

    ## 해결 요함
    T_s2s_left = np.eye(4)
    T_s2s_left[:3, :3] = get_rotation_matrix(np.deg2rad(30), np.deg2rad(90), 0).T
    T_s2s_left[0, 3] = 0
    T_s2s_left[1, 3] = 0
    T_s2s_left[2, 3] = 0
    # print(T_s2o_left)
    T_s2o_left = T_s2s_left.T * T_s2o_front

    ## 해결 요함
    T_s2s_right = np.eye(4)
    T_s2s_right[:3, :3] = get_rotation_matrix(np.deg2rad(-30), np.deg2rad(90), 0).T
    T_s2s_right[0, 3] = 0
    T_s2s_right[1, 3] = 0
    T_s2s_right[2, 3] = 0
    # print(T_s2o_right)
    T_s2o_right = T_s2s_right * T_s2o_front

    # pcd_front = PROCESS_3D_PCD(obj_front.MATCHED_LIST_PCD, obj_front.MATCHED_LIST_ODOM, T_s2o_front)
    # front_3d_npy = pcd_front.get_processed_3d_npy()

    # pcd_left = PROCESS_3D_PCD(obj_left.MATCHED_LIST_PCD, obj_left.MATCHED_LIST_ODOM, T_s2o_left)
    # left_3d_npy = pcd_left.get_processed_3d_npy()

    pcd_right = PROCESS_3D_PCD(obj_right.MATCHED_LIST_PCD, obj_right.MATCHED_LIST_ODOM,  T_s2o_right)
    right_3d_npy = pcd_right.get_processed_3d_npy()


    accumulated_pcd_dir = os.path.join(dataset_path, 'accumulated.pcd')
    accumulated_cloud = deepcopy(right_3d_npy)
    # accumulated_cloud = np.vstack((accumulated_cloud, right_3d_npy))
    # accumulated_cloud = np.vstack((accumulated_cloud, left_3d_npy))

    # accum_pcd = o3d.geometry.PointCloud()
    # accum_pcd.points = o3d.utility.Vector3dVector(accumulated_cloud)
    # o3d.visualization.draw_geometries([accum_pcd])

    accumulated_pcd_dir = os.path.join(dataset_path, 'test.pcd')
    save_pcd_from_npy(accumulated_cloud, accumulated_pcd_dir)