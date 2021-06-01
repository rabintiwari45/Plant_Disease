!git clone https://github.com/tensorflow/models.git
%cd /content/models/research/
!protoc object_detection/protos/*.proto --python_out=.


import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from google.colab.patches import cv2_imshow
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


PATH_TO_SAVED_MODEL = "/content/drive/MyDrive/plantdoc/PlantDoc-Dataset/exported_model/saved_model"
PATH_TO_LABELS = "/content/drive/My Drive/data_tfrecord/PlantDoc.v1-resize-416x416.tfrecord/train/leaves_label_map.pbtxt"
IMAGE_PATH = "/content/drive/MyDrive/plantdoc/PlantDoc-Dataset/train/Apple Scab Leaf/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"


def object_model_predict(img_path,PATH_TO_SAVED_MODEL,PATH_TO_LABELS):
    detect_fn = tf.compat.v2.saved_model.load(str(PATH_TO_SAVED_MODEL),None)
    category_index = label_map_util.create_category_index_from_labelmap(str(PATH_TO_LABELS),use_display_name=True)
    image_np = plt.imread(img_path)
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.5,
          agnostic_mode=False)  
    
    return image_np_with_detections

if __name__ == '__main__':
  image = object_model_predict(IMAGE_PATH,PATH_TO_SAVED_MODEL,PATH_TO_LABELS)
  cv2_imshow(image)