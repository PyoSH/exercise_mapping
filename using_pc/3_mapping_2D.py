#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:49:17 2024

@author: pyo
"""
from utils_mapping import *
from data_processing import *
import os 
from copy import deepcopy

dataset_path = '/home/pyo/b1_perception/exercise_mapping'
pcd_path = os.path.join(dataset_path, 'test.pcd')
processed_pcd_path = os.path.join(dataset_path, 'processed.pcd')

if __name__ =='__main__':
    
    ### 1. load accumulated pcd
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    ### 2.data denoise and downsampling
    ## 2.1 denoise
    cl, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    denoised_pcd = pcd.select_by_index(idx)
    
    ## 2.2 downsampling
    cp_pcd = deepcopy(denoised_pcd)
    down_pcd = cp_pcd.voxel_down_sample(voxel_size=0.1)
    
    ## 2.3 crop Z
    np_pcd = np.array(cp_pcd.points)
    croped_np_pcd = np_pcd[(np_pcd[:,2] >= 0.2) & (np_pcd[:,2] <= 0.6)]

    projected_np_pcd = croped_np_pcd[:, :2]
    # fit_wall(projected_np_pcd)

    ogd = make_2D_OGD(projected_np_pcd)
    # plt.imshow(ogd, origin='lower', cmap='gray')
    # plt.show()

    croped_pcd = o3d.geometry.PointCloud()
    croped_pcd.points = o3d.utility.Vector3dVector(croped_np_pcd)
    # o3d.visualization.draw_geometries([down_pcd])
    o3d.io.write_point_cloud(processed_pcd_path, down_pcd)
