#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:51:36 2024

@author: pyo
"""

import os
import numpy as np
import open3d as o3d
import math


def npy_from_pcd(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    
    return points   

def save_pcd_from_npy(npy, pcd_file_dir):
    # Numpy data -> open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy)
    
    # save as pcd form
    o3d.io.write_point_cloud(pcd_file_dir, pcd)
    print(f'PCD file saved as {pcd_file_dir}')

    
def get_rotation_matrix(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    return np.dot(Rz, np.dot(Ry, Rx))

def get_euler_from_quaternion(q_x, q_y, q_z, q_w):
    # 롤 (Roll)
    t0 = +2.0 * (q_w * q_y + q_x * q_z)
    t1 = +1.0 - 2.0 * (q_y * q_y + q_x * q_x)
    roll_x = math.atan2(t0, t1)
    
    # 피치 (Pitch)
    t2 = +2.0 * (q_w * q_x - q_y * q_z)
    t2 = +1.0 if t2 > +1.0 else t2  # 수치적 안정성을 위해 클램핑
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    # 요 (Yaw)
    t3 = +2.0 * (q_w * q_z + q_x * q_y)
    t4 = +1.0 - 2.0 * (q_x * q_x + q_z * q_z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z  # 라디안 단위로 반환

def get_transform_pose(curr):
    x = curr[0]
    y = curr[1]
    z = curr[2]
    roll_x, pitch_y, yaw_z = get_euler_from_quaternion(curr[3], curr[4], curr[5], curr[6])
    
    retval = np.eye(4)
    retval[:3, :3] = get_rotation_matrix(roll_x, pitch_y, yaw_z)
    retval[0, 3] = x
    retval[1, 3] = y
    retval[2, 3] = z
    
    return retval

def get_transform_between_poses(prev, curr):
    prev_trsf = get_transform_pose(prev)
    curr_trsf = get_transform_pose(curr)
    
    return np.dot(np.linalg.inv(prev_trsf), curr_trsf)

def apply_transform2cloud(cloud, trsf):
    homogeneous_points = np.hstack((cloud, np.ones((cloud.shape[0], 1))))
    applied_points = np.dot(trsf, homogeneous_points.T).T
    
    return applied_points[:, :3]
