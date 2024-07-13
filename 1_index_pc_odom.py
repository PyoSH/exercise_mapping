#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur July 11 16:31:05 2024

@author: PyoSH
"""

import os
import numpy as np
import open3d as o3d
import math

EXTENSIONS = ['.pcd']

def file_path(root, filename):
    return os.path.join(root, '{}'.format(filename) )

def is_pcd(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def binary_search_closest(arr, target):
    low, high = 0, len(arr) - 1
    closest_num = None
    
    # 배열이 빈 경우 바로 반환
    if len(arr) == 0:
        return None
    # 배열에 하나의 요소만 있는 경우 바로 반환
    if len(arr) == 1:
        return arr[0]

    while low <= high:
        mid = (low + high) // 2

        # 가능한 가장 가까운 숫자를 업데이트
        if closest_num is None or abs(arr[mid] - target) < abs(closest_num - target):
            closest_num = arr[mid]

        if arr[mid] < target:
            low = mid + 1
        elif arr[mid] > target:
            high = mid - 1
        else:
            return arr[mid]  # 정확히 일치하는 값 찾음

    return closest_num


dataset_path = '/home/pyo/b1_perception'
pcd_path = os.path.join(dataset_path, 'pcds')
odom_path = dataset_path

### 1. Load lists
# pcds
pcd_filenames = [file_path(pcd_path, f) for f in os.listdir(pcd_path) if is_pcd(f)]
pcd_filenames.sort()

# load odom & timeStamps
odom_origin = np.loadtxt(os.path.join(odom_path, 'odometry.txt'),delimiter=',')
odom_timestamps = list(np.array(odom_origin[:,-1]))
pc_timestamps = list(np.loadtxt(os.path.join(pcd_path, "pc_timestamp.txt"),delimiter=',')[:,1])

# collect time-matched poses which related in pointcloud
odom_selected= []

for i in pc_timestamps:
    curr_pc_pub_time = i
    closest_item = binary_search_closest(odom_timestamps, curr_pc_pub_time)
    closest_idx = odom_timestamps.index(closest_item)
    odom_selected.append(odom_origin[closest_idx, :])

arr_odom_selected = np.array(odom_selected)
saveDir= os.path.join(odom_path, "odom_matched.txt")
np.savetxt(saveDir, arr_odom_selected, fmt="%f", delimiter=',')     
print(f"Data at {saveDir} saved.")   