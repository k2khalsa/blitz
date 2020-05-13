# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:24:58 2020

@author: karan
"""
import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

#Camera set up
 
while True:
       frames = pipeline.wait_for_frames()
       depth_frame = frames.get_depth_frame()
       if not depth_frame: 
           continue
       width = depth_frame.get_width()
       height = depth_frame.get_height()
       #print(width,height)
        
       #Calculate distance
       dist_to_center = depth_frame.get_distance(int(width/2), int(height/2))
       print('The camera is facing an object:',dist_to_center,'meters away')