#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 00:21:08 2024

@author: pyo
"""

import numpy as np
import open3d as o3d
import math

def filter_distance(pc, max_dist=10.0):
    distances = np.linalg.norm(pc, axis=1)
    filtered_cloud = pc[distances <= max_dist]
    
    return filtered_cloud

