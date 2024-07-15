#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 00:21:08 2024

@author: pyo
"""

import numpy as np
import open3d as o3d
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

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
    print(curr_norm, thres)
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

def make_2D_OGD(pc_2d):
    # 점유 격자 지도의 해상도 및 크기 설정
    resolution = 0.05  # 5cm 당 하나의 격자
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