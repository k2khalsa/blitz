#!/usr/bin/env python
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
tf.gfile = tf.io.gfile
import zipfile
import imageio

from datetime import datetime
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
cap = cv2.VideoCapture(0)


sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from utils import label_map_util

from utils import visualization_utils_position as vis_util


# This is needed to display the images.
get_ipython().run_line_magic('matplotlib', 'inline')

# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[ ]:


# What model to download.

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model

# In[ ]:


opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_cl)
#category_index = label_map_util.create_category_index(categories)



# In[ ]:

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:    
            ret, image_np = cap.read()

  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_tensor= detection_graph.get_tensor_by_name('image_tensor:0')
            
            boxes= detection_graph.get_tensor_by_name('detection_boxes:0')
            
            scores=detection_graph.get_tensor_by_name('detection_scores:0')
            
            classes=detection_graph.get_tensor_by_name('detection_classes:0')
            
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
  
    
  #loop through video frames
            input_video= 'traffic'
            video_reader=imageio.get_reader('%s.mp4' %input_video)
            video_writer=imageio.get_writer('%s_annotated.mp4' %input_video, fps=10)
            t0=datetime.now()
            n_frames=0
            for frame in video_reader:
                image_np=frame
                n_frames+=1
                image_np_expanded=np.expand_dims(image_np,axis=0)
      # Actual detection.
                (boxes, scores, classes, num_detection)= sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor:image_np_expanded})
      # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=8)
                
                video_writer.append_data(image_np)
            fps=n_frames/(datetime.now()-t0).total_seconds()
            print("frames processed: %s.speed:%s fps"%(n_frames,fps))
            
            video_writer.close()

