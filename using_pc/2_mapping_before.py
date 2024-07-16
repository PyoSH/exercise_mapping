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
    pcd_path = os.path.join(dataset_path, 'pcds')
    odom_path = dataset_path

    # ### 1. Load lists
    obj0 = MATCH_PC_ODOM(dataset_path, pcd_path, odom_path)
    obj0.run()

    ### transform clouds with (sensor -> odom) coordinate frame T,
    T_sensor2odom = np.eye(4)
    T_sensor2odom[:3, :3] = get_rotation_matrix(np.deg2rad(90), np.deg2rad(-90), 0).T
    T_sensor2odom[0, 3] = 0
    T_sensor2odom[1, 3] = 0
    T_sensor2odom[2, 3] = 0

    pcd_front = PROCESS_3D_PCD(obj0.MATCHED_LIST_PCD, obj0.MATCHED_LIST_ODOM, T_sensor2odom)
    front_3d_npy = pcd_front.get_processed_3d_npy()

    accumulated_pcd_dir = os.path.join(dataset_path, 'accumulated.pcd')
    save_pcd_from_npy(front_3d_npy, accumulated_pcd_dir)