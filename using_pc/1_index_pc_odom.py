#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur July 11 16:31:05 2024

@author: PyoSH
"""

from utils_mapping import *

if __name__ =='__main__':

    dataset_path = '/home/pyo/b1_perception/exercise_mapping'
    pcd_path = os.path.join(dataset_path, 'pcds_front')
    odom_path = dataset_path

    test = MATCH_PC_ODOM(dataset_path, pcd_path, odom_path)
    test.run()
    test.save_txt_matched()

    # odom_origin = np.loadtxt(os.path.join(odom_path, 'odometry.txt'), delimiter=',')
