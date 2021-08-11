#!/usr/bin/env python

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import argparse

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import cv2

# This is needed since the current program is in the object_detection folder.
sys.path.append("..")
#from object_detection.utils import ops as utils_ops
from utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from utils import label_map_util

from utils import visualization_utils as vis_util
THRESHOLD = 0.4
# What model to use
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
#MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
#MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

#print(MODEL_NAME)
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# If model does not exist locally then download it
if not os.path.exists(PATH_TO_FROZEN_GRAPH):
  print("Downloading Model")
  opener = urllib.request.URLopener()
  opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
  tar_file = tarfile.open(MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
def drawBoxesOnImage(img, output_dict, category_index, use_normalized_coordinates=True,line_thickness=2):
  scores = output_dict['detection_scores']
  boxes = output_dict['detection_boxes']
  classes = output_dict['detection_classes']
  a = output_dict['detection_scores']
  length = a.size
  num_boxes = 5
  global THRESHOLD
  
  for i in range(min(length, num_boxes)):
    score = scores[i]
    box = boxes[i]
    category = classes[i]
    #print(category)
    if(category ==1 and score >= THRESHOLD):
      #draw box
      width = img.shape[1]
      height = img.shape[0]
      x1=int(box[1] *width)
      y1 =int(box[0]*height)
      x2 = int(box[3] * width)
      y2 = int(box[2]*height)
      cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),line_thickness)
      text =  str((int(score*100))) + "%"
      cv2.putText(img, text, (x1+20,y1+25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)









  #cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
  # vis_util.visualize_boxes_and_labels_on_image_array(
  #         image_np,
  #         output_dict['detection_boxes'],
  #         output_dict['detection_classes'],
  #         output_dict['detection_scores'],
  #         category_index,
  #         instance_masks=output_dict.get('detection_masks'),
  #         use_normalized_coordinates=use_normalized_coordinates,
  #         line_thickness=line_thickness)

def run_inference_for_video(cap, graph):
  i =0
  with graph.as_default():
    with tf.Session() as sess:
      while True:
        # Read next image frame from video
        ret, image_np = cap.read()
        if not ret:
          cv2.destroyAllWindows()
          break
        # Convert image from default opencv BGR format to RGB format required by tensorflow model
        image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) 
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image = np.expand_dims(image_np, axis=0)
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
          tensor_name = key + ':0'
          if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
        if 'detection_masks' in tensor_dict:
          # The following processing is only for single image
          detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
          detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
          # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
          real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
          detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
          detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
          detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
          detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
          # Follow the convention by adding back the batch dimension
          tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: image})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
          output_dict['detection_masks'] = output_dict['detection_masks'][0]
        # Visualization of the results of a detection.
        # vis_util.visualize_boxes_and_labels_on_image_array(
        #   image_np,
        #   output_dict['detection_boxes'],
        #   output_dict['detection_classes'],
        #   output_dict['detection_scores'],
        #   category_index,
        #   instance_masks=output_dict.get('detection_masks'),
        #   use_normalized_coordinates=True,
        #   line_thickness=8)
        
        if(i == 0):
          print(output_dict)
          i+=1
        drawBoxesOnImage(image_np,
          output_dict,
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

        # Show frame on screen
        cv2.imshow('object detection: '+MODEL_NAME, cv2.resize(image_np, (800,600)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break

  
        
def main(input):
  # Capture video from input (input 0=Webcam)
  cap = cv2.VideoCapture(input)
  #cap.set(cv2.CAP_PROP_FPS, 10) # Set video capture to 10fps
  run_inference_for_video(cap, detection_graph)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input', help='Path to input video', nargs='?', default=0) # webcam is default input
  #parser.add_argument('-i', '--input', help='Path to input video', default=0) # webcam is default input
  args = parser.parse_args()
  main(args.input)
