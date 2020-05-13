#!/usr/bin/env python
# coding: utf-8
# In[52]:
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
tf.gfile = tf.io.gfile
import pyrealsense2 as rs
import zipfile
import pandas as pd
import datetime
import time

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
cap = cv2.VideoCapture(1)

output_directory = 'C:/Users/karan/Desktop/PreceptionTrial/models/research/object_detection/results'

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# In[54]:

from utils import label_map_util

from utils import visualization_utils as vis_util

# In[55]:


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


# In[56]:


opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# In[57]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# In[58]:


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# In[59]:


def run_inference_for_single_image(image, graph):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


# In[60]:


last_position_time = datetime.datetime.now()
time.sleep(10)
new_position_time = datetime.datetime.now()

os.makedirs(output_directory, exist_ok=True)

if os.path.exists(output_directory+'/results.csv'):
    df = pd.read_csv(output_directory+'/results.csv')
else:
    df = pd.DataFrame(columns=['timestamp', 'x_min', 'x_max', 'y_min', 'y_max'])
try:        
    with detection_graph.as_default():
        with tf.compat.v1.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.compat.v1.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                      tensor_name)
                        
                #realsense cam intialization
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, 800, 600, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 800, 600, rs.format.bgr8, 30)
                profile = pipeline.start(config)
                while True:
                    frames = pipeline.wait_for_frames()
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not depth_frame or not color_frame:
                        continue
                    
                    ret, image_np = cap.read()
                    image_height, image_width, _ = image_np.shape
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    output_dict = run_inference_for_single_image(image_np, detection_graph)
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                    
                    # Get data(label, xmin, ymin, xmax, ymax, depth)
                    output = []
                    for index, score in enumerate(output_dict['detection_scores']):
                            new_position_time = datetime.datetime.now()
                            time_delta = (new_position_time-last_position_time).total_seconds()
                            if (score < 0.75 or time_delta < 10):
                                continue                            
                            label = category_index[output_dict['detection_classes'][index]]['name']
                            ymin, xmin, ymax, xmax = output_dict['detection_boxes'][index]
                            
                            #depth calculator
                            depth = np.asanyarray(depth_frame)
                            depth = depth[(int(xmin * image_width)):(int(xmax * image_width)),
                                          (int(ymin * image_height)):(int(ymax * image_height))].astype(float)
                            
                            # Get data scale from the device and convert to meters
                            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                            depth = depth * depth_scale
                            dist,_,_,_ = cv2.mean(depth)
                            
                            output.append((label, int(xmin * image_width), int(ymin * image_height), 
                                           int(xmax * image_width), int(ymax * image_height), depth))
                            
                    # Save incident (could be extended to send a email or something)
                    for l, x_min, y_min, x_max, y_max, depth in output:
                        if l == 'person':
                            last_position_time = datetime.datetime.now()
                            df.loc[len(df)] = [datetime.datetime.now().time(), x_min, x_max, y_min, y_max, depth]
                            df.to_csv(output_directory+'/results.csv', index=None)
                            
                    
except Exception as e:
    print(e)
    cap.release()

