#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 00:21:08 2024

@author: pyo
"""

import numpy as np
import open3d as o3d
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from utils_mapping import apply_transform2cloud, get_transform_between_poses

def filter_distance(pc, max_dist=10.0):
    distances = np.linalg.norm(pc, axis=1)
    filtered_cloud = pc[distances <= max_dist]
    
    return filtered_cloud

def get_anomaly_thres(pcds):
    list_norm = []
    for i in range(len(pcds)):
        # temp_norm = np.linalg.norm(pcds[i], )
        temp_norm = np.linalg.norm(pcds[i], np.inf)
        list_norm.append(temp_norm)

    retval = np.median(np.array(list_norm))

    return retval

def is_anomaly_norm(pc, thres):
    curr_norm = np.linalg.norm(pc, np.inf)
    # print(curr_norm, thres)
    if (curr_norm > thres):
        return True
    else:
        return False

def segment_ground(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, 
                                             ransac_n=3, num_iterations=1000)
    ground = pcd.select_by_index(inliers)
    non_ground = pcd.select_by_index(inliers, invert=True)
    
    return ground, non_ground

def make_2D_OGD(pc_2d, resol=0.05):
    # 점유 격자 지도의 해상도 및 크기 설정
    resolution = resol  # 5cm 당 하나의 격자
    x_max, y_max = pc_2d.max(axis=0)
    x_min, y_min = pc_2d.min(axis=0)

    # 격자 크기 계산
    x_size = int((x_max - x_min) / resolution) + 1
    y_size = int((y_max - y_min) / resolution) + 1

    # 격자 초기화
    grid = np.zeros((x_size, y_size), dtype=np.uint8)

    # 포인트를 격자에 매핑
    for x, y in pc_2d:
        grid_idx_x = int((x - x_min) / resolution)
        grid_idx_y = int((y - y_min) / resolution)
        grid[grid_idx_x, grid_idx_y] = 1  # 점유 상태로 설정

    return grid

def fit_wall(pc_2d):
    ransac = RANSACRegressor()

    x = pc_2d[:, 0].reshape(-1, 1)
    y = pc_2d[:, 1]
    ransac.fit(x, y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_x = np.arange(x.min(), x.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_x)

    plt.scatter(x[inlier_mask], y[inlier_mask], color='yellow', marker='.', label='Inliers')
    plt.scatter(x[outlier_mask], y[outlier_mask], color='red', marker='.', label='Outliers')
    plt.plot(line_x, line_y_ransac, color='blue', linewidth=2, label='RANSAC regressor')
    plt.gca().invert_yaxis()  # y축 방향 반전 (이미지 좌표에 맞춤)
    plt.legend()
    plt.show()

class PROCESS_3D_PCD():
    def __init__(self, matched_pcd, matched_odom, T):
        self.PCD = matched_pcd
        self.ODOM = matched_odom
        self.T = T

    def get_processed_3d_npy(self):
        # sensor_clouds = deepcopy(self.PCD)
        sensor_clouds = self.PCD

        for i in range(len(self.PCD)):
            sensor_clouds[i] = apply_transform2cloud(sensor_clouds[i], self.T)

        ### 3. transform clouds from poses & accumulate clouds
        accumulated_cloud = None
        cnt = 0
        accum_cnt = 0
        is_accum_empty = True

        ## 3.1 detect anomaly point cloud
        anomaly_thres = get_anomaly_thres(sensor_clouds)

        ## 3.2 accumulate point cloud
        for i in range(len(sensor_clouds)):
            if (is_anomaly_norm(sensor_clouds[i], anomaly_thres)):
                pass
            else:
                if (is_accum_empty):
                    accumulated_cloud = sensor_clouds[i]
                    is_accum_empty = False
                else:
                    filtered_cloud = filter_distance(sensor_clouds[i], 10.0)

                    prev_pose = self.ODOM[0][0:7]
                    curr_pose = self.ODOM[i][0:7]

                    T_poses = get_transform_between_poses(prev_pose, curr_pose)

                    applied_cloud = apply_transform2cloud(filtered_cloud, T_poses)

                    accumulated_cloud = np.vstack((accumulated_cloud, applied_cloud))
                    np.vstack((accumulated_cloud, applied_cloud))
                    accum_cnt = accum_cnt + accumulated_cloud.shape[0]

            print(f'{cnt}th cloud shape: {sensor_clouds[i].shape[0]} | ACCUMULATED clouds : {accum_cnt}')
            cnt = cnt + 1

        return accumulated_cloud