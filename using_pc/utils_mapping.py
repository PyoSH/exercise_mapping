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
import rosbag
from sensor_msgs import point_cloud2
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

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

    retval = np.dot(Rz, np.dot(Ry, Rx))
    # return retval
    return Rz @ Ry @ Rx

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

# def get_R_from_quaternion(q_x, q_y, q_z, q_w):
#     R.from_quat()

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

def file_path(path, fileName):
    return os.path.join(path, '{}'.format(fileName) )

def is_pcd(fileName):
    EXTENSIONS = ['.pcd']
    return any(fileName.endswith(ext) for ext in EXTENSIONS)

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

class MATCH_PC_ODOM():
    def __init__(self, root_dir, pcd_path, odom_path):
        self.ROOT_DIR = root_dir
        self.PCD_PATH = pcd_path
        self.ODOM_PATH = odom_path

        self.MATCHED_LIST_PCD = []
        self.MATCHED_LIST_ODOM = []


    def run(self):
        ### 1. Load lists
        # pcds
        pcd_filenames = [file_path(self.PCD_PATH, f) for f in os.listdir(self.PCD_PATH) if is_pcd(f)]
        pcd_filenames.sort()

        for pcd_file in pcd_filenames:
            self.MATCHED_LIST_PCD.append(npy_from_pcd(pcd_file))

        # load odom & timeStamps
        odom_origin = np.loadtxt(os.path.join(self.ODOM_PATH, 'odometry.txt'), delimiter=',')
        odom_timestamps = list(np.array(odom_origin[:, -1]))
        pc_timestamps = list(np.loadtxt(os.path.join(self.PCD_PATH, "pc_timestamp.txt"), delimiter=',')[:, 1])

        # collect time-matched poses which related in pointcloud
        for i in pc_timestamps:
            curr_pc_pub_time = i
            closest_item = binary_search_closest(odom_timestamps, curr_pc_pub_time)
            closest_idx = odom_timestamps.index(closest_item)
            self.MATCHED_LIST_ODOM.append(odom_origin[closest_idx, :])


    def save_txt_matched(self):
        arr_odom_selected = np.array(self.MATCHED_LIST_ODOM)
        saveDir = os.path.join(self.ODOM_PATH, "odom_matched.txt")
        np.savetxt(saveDir, arr_odom_selected, fmt="%f", delimiter=',')
        print(f"Data at {saveDir} saved.")


class PCL_FROM_ROS():
    def __init__(self, root_dir, bag_path, topic_name, direction, timestamp_path):
        self.ROOT_DIR = root_dir
        self.BAG_PATH = bag_path
        self.TOPIC_NAME = topic_name
        self.TIMESTAMP_PATH = timestamp_path
        self.DIRECTION = direction
        self.PC_NPY_LIST = []
        self.TIMESTAMP_LIST = []

    def run(self):
        bag = rosbag.Bag(self.BAG_PATH, 'r')
        count = 0
        time_0 = None
        is_1st = True

        with open(self.TIMESTAMP_PATH, 'w') as timestamp_txt:
            for topic, msg, t_ in bag.read_messages(topics=[self.TOPIC_NAME]):
                _timestamp = msg.header
                ts = _timestamp.stamp
                t = ts.secs + ts.nsecs / float(1e9)

                if (is_1st):
                    time_0 = t
                    is_1st = False
                else:
                    pass

                t = t - time_0

                pc = point_cloud2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
                pcloud = o3d.geometry.PointCloud()
                pcloud.points = o3d.utility.Vector3dVector(list(pc))

                # PCD 파일로 저장
                if (msg):
                    pcd_filename = os.path.join(self.ROOT_DIR, self.DIRECTION , f"output_{count}.pcd")
                    o3d.io.write_point_cloud(pcd_filename, pcloud)
                    print(f"Saved {pcd_filename}")

                    timestamp_txt.write(f"{int(msg.header.seq)}, {'{:.12f}'.format(t)}\n")
                    self.TIMESTAMP_LIST.append(t)
                    self.PC_NPY_LIST.append(np.asarray(pcloud.points))

                    count += 1

        bag.close()
        print(f"Total {count} point clouds saved.")

class ODOM_FROM_ROS():
    def __init__(self, root_dir, bag_path, topic_name, output_dir):
        self.ROOT_DIR =root_dir
        self.BAG_PATH = bag_path
        self.TOPIC_NAME = topic_name
        self.OUTPUT = output_dir

    def save_odometry_to_txt(self):
        bag = rosbag.Bag(self.BAG_PATH, 'r')
        count = 0
        infoVector=np.zeros(9)
        time_0 = 0
        pose_0 = None
        pose_0_orientation_inv = None
        is_1st = True

        with open(self.OUTPUT, 'w') as file:
            for _, msg, t_ in bag.read_messages(topics=[self.TOPIC_NAME]):
                # 오도메트리 데이터 추출
                _pose=msg.pose.pose
                _timestamp =msg.header

                ts = _timestamp.stamp
                t = ts.secs + ts.nsecs / float(1e9)

                # 1st iteration -> set t_0
                if(is_1st):
                    time_0 = t
                    pose_0 = deepcopy(_pose)
                    is_1st = False
                else:
                    pass

                t = t - time_0
                curr_orientation =[_pose.orientation.x, _pose.orientation.y, _pose.orientation.z, _pose.orientation.w]

                infoVector[0] = _pose.position.x
                infoVector[1] = _pose.position.y
                infoVector[2] = _pose.position.z
                infoVector[3] = curr_orientation[0]
                infoVector[4] = curr_orientation[1]
                infoVector[5] = curr_orientation[2]
                infoVector[6] = curr_orientation[3]
                infoVector[7] = int(_timestamp.seq)
                infoVector[8] = '{:.12f}'.format(t)

                if(infoVector[0]):
                    cur_pose=infoVector
                    file.write(str(cur_pose[0])+','+str(cur_pose[1])+','+str(cur_pose[2])+','
                        +str(cur_pose[3])+','+str(cur_pose[4])+','+str(cur_pose[5])+','+str(cur_pose[6])+','
                        +str(cur_pose[7])+','+str(cur_pose[8])+'\n')

                    count += 1
                    print(f"Data at {count} saved.")
                else:
                    print(f"No data")
                    pass

        bag.close()
        print(f"Total {count} odometry entries saved to {self.OUTPUT}.")