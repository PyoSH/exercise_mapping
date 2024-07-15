#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 22:05:01 2024

@author: pyo
"""
import numpy as np

from utils_mapping import *
from data_processing import *
import os
from copy import deepcopy

def file_path(root, filename):
    return os.path.join(root, '{}'.format(filename) )

def is_pcd(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


dataset_path = '/home/pyo/b1_perception/exercise_mapping'
pcd_path = os.path.join(dataset_path, 'pcds')
odom_path = dataset_path
EXTENSIONS = ['.pcd']

if __name__ =='__main__':
    
    ### 1. Load lists
    # load pcds
    pcd_filenames = [file_path(pcd_path, f) for f in os.listdir(pcd_path) if is_pcd(f)]
    pcd_filenames.sort()

    # load odom 
    odom = list(np.loadtxt(os.path.join(odom_path, 'odom_matched.txt'),delimiter=','))
    
    ### 2. prepare clouds in list data structure (pcd form -> npy form)
    sensor_clouds = []
    
    for pcd_file in pcd_filenames:
        sensor_clouds.append(npy_from_pcd(pcd_file))
        
    ### 2.5 transform clouds with (sensor -> odom) coordinate frame T, 
    ## CAUTION : watch out your system!!
    
    T_sensor2odom = np.eye(4)
    T_sensor2odom[:3, :3] = get_rotation_matrix(np.deg2rad(90), np.deg2rad(-90), 0).T
    T_sensor2odom[0, 3] = 0
    T_sensor2odom[1, 3] = 0
    T_sensor2odom[2, 3] = 0
    
    odom_clouds = deepcopy(sensor_clouds)
    
    for i in range(len(odom_clouds)):
        odom_clouds[i] = apply_transform2cloud(odom_clouds[i], T_sensor2odom)
        
    ### 3. transform clouds from poses & accumulate clouds    
    accumulated_cloud = None
    cnt = 0
    accum_cnt = 0

    ## 3.1 detect anomaly point cloud
    anomaly_thres = get_anomaly_thres(odom_clouds)

    ## 3.2 accumulate point cloud
    for i in range(len(odom_clouds)):
        if (is_anomaly_norm(odom_clouds[i], anomaly_thres)):
            pass
        else:
            if(i == 0):
                accumulated_cloud = odom_clouds[i]
            else:
                filtered_cloud = filter_distance(odom_clouds[i], 10.0)

                prev_pose = odom[0][0:7]
                curr_pose = odom[i][0:7]

                T = get_transform_between_poses(prev_pose, curr_pose)

                applied_cloud = apply_transform2cloud(filtered_cloud, T)

                accumulated_cloud = np.vstack((accumulated_cloud, applied_cloud))
                accum_cnt = accum_cnt + accumulated_cloud.shape[0]

        print(f'{cnt}th cloud shape: {odom_clouds[i].shape[0]} | ACCUMULATED clouds : {accum_cnt}')
        cnt = cnt +1

    ### 4. save file (accumulated cloud npy -> pcd)
    accumulated_pcd_dir = os.path.join(dataset_path, 'accumulated.pcd')
    save_pcd_from_npy(accumulated_cloud, accumulated_pcd_dir)
