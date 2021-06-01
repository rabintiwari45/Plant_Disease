from flask import Flask,render_template
from flask import request
import os
import numpy as np

import sys
import tensorflow as tf


from tensorflow.python.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


from PIL import Image
import matplotlib.pyplot as plt
import warnings
import pathlib
#import cv2

app = Flask(__name__)

class_label = ['Apple Scab Leaf','Apple leaf','Apple rust leaf','Bell_pepper leaf',
               'Bell_pepper leaf spot','Blueberry leaf','Cherry leaf','Corn Gray leaf spot',
               'Corn leaf blight','Corn rust leaf','Peach leaf','Potato leaf early blight',
               'Potato leaf late blight','Raspberry leaf','Soyabean leaf','Squash Powdery mildew leaf',
               'Strawberry leaf','Tomato Early blight leaf','Tomato Septoria leaf spot','Tomato leaf',
               'Tomato leaf bacterial spot','Tomato leaf late blight','Tomato leaf mosaic virus',
               'Tomato leaf yellow virus','Tomato mold leaf','grape leaf','grape leaf black rot'
              ]


UPLOAD_FOLDER = "C:/Users/Rabin/Desktop/Image_classification/static"

#MODEL_PATH = 'model.h5'
model = load_model('model.h5')
PATH_TO_SAVED_MODEL = "saved_model"
PATH_TO_LABELS = "leaves_label_map.pbtxt"

def model_predict(img_path,model):
    img = img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    pred = model.predict(img_array)
    pred = np.argmax(pred,axis=1)
    return pred

def object_model_predict(img_path,PATH_TO_SAVED_MODEL,PATH_TO_LABELS):
    detect_fn = tf.compat.v2.saved_model.load(str(PATH_TO_SAVED_MODEL),None)
    category_index = label_map_util.create_category_index_from_labelmap(str(PATH_TO_LABELS),use_display_name=True)
    image_np = plt.imread(img_path)
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    #input_tensor = tf.convert_to_tensor(image_np)
    print(input_tensor.shape)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    print(input_tensor.shape)
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
          min_score_thresh=.4,
          agnostic_mode=False)  

    plt.figure()
    plt.figure()
    plt.imshow(image_np_with_detections)
    plt.show()
    print(image_np_with_detections.shape)
    #cv2.imshow(image_np_with_detections)
    return image_np_with_detections #plt.imshow(image_np_with_detections)

@app.route('/')
def home():
   return render_template('index.html')


@app.route('/',methods=['GET','POST'])

def DetectImage():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            preds = model_predict(image_location,model)
            #preds_object = object_model_predict(image_location,PATH_TO_SAVED_MODEL,PATH_TO_LABELS)
            #im = Image.fromarray(preds_object)
            #im = Image.fromarray((preds_object * 255).astype(np.uint8))
            #im.save('static/upload/' + image_file.filename)
            #print(preds_object) 

            return render_template("index.html",prediction=class_label[int(preds)])#,image_loc=image_file.filename)
            #return render_template("index.html",image_loc=image_file.filename)

    return render_template('index.html',prediction=None,image_loc=None)





if __name__ == '__main__':
    app.run(port = 2000,debug=True)
