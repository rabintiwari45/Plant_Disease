import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from IPython.display import Image

MODEL_PATH = '/content/drive/MyDrive/plantdoc/PlantDoc-Dataset/model_mobilenet_6525.h5'
IMAGE_PATH = '/content/drive/MyDrive/plantdoc/PlantDoc-Dataset/test/Apple Scab Leaf/052609%20Hartman%20Crabapple%20scab%20single%20leaf.JPG.jpg'
IMAGE_SIZE = (224,224)

last_conv_layer_name = "conv_pw_13_relu"
classifier_layer_names = [
                          "global_average_pooling2d",
                          "fc1",
                          "Dropout_1",
                          "batch_normalization",
                          "fc2",
                          "Dropout",
                          "output",        
                          ]

class_label = ['Apple Scab Leaf','Apple leaf','Apple rust leaf','Bell_pepper leaf',
               'Bell_pepper leaf spot','Blueberry leaf','Cherry leaf','Corn Gray leaf spot',
               'Corn leaf blight','Corn rust leaf','Peach leaf','Potato leaf early blight',
               'Potato leaf late blight','Raspberry leaf','Soyabean leaf','Squash Powdery mildew leaf',
               'Strawberry leaf','Tomato Early blight leaf','Tomato Septoria leaf spot','Tomato leaf',
               'Tomato leaf bacterial spot','Tomato leaf late blight','Tomato leaf mosaic virus',
               'Tomato leaf yellow virus','Tomato mold leaf','grape leaf','grape leaf black rot'
              ]

def get_model(model_path):
  return load_model(model_path)

def get_image(image_path):
  image = keras.preprocessing.image.load_img(image_path)
  image = keras.preprocessing.image.img_to_array(image)
  image = cv2.resize(image,IMAGE_SIZE)
  return image


def get_img_array(img_path,size):
  img = keras.preprocessing.image.load_img(img_path,target_size=size)
  array = keras.preprocessing.image.img_to_array(img)
  array = np.expand_dims(array,axis=0)
  array = array/255.0
  return array

def make_gradcam_heatmap(img_array,model,last_conv_layer_name,classifier_layer_names):
  last_conv_layer = model.get_layer(last_conv_layer_name)
  last_conv_layer_model = keras.Model(model.inputs,last_conv_layer.output)

  classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
  x = classifier_input
  for layer_name in classifier_layer_names:
    x = model.get_layer(layer_name)(x)
  classifier_model = keras.Model(classifier_input,x)

  with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_array)
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    #top_pred_index = 12
    top_class_channel = preds[:,top_pred_index]
    print(top_class_channel)
  grads = tape.gradient(top_class_channel,last_conv_layer_output)

  pooled_grads = tf.reduce_mean(grads,axis=(0,1,2))

  last_conv_layer_output = last_conv_layer_output.numpy()[0]
  pooled_grads = pooled_grads.numpy()
  for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:,:,i] *= pooled_grads[i]
  heatmap = np.mean(last_conv_layer_output,axis=-1)

  heatmap = np.maximum(heatmap,0)/np.max(heatmap)
  return heatmap,top_pred_index

def display(alpha=0.8):
  image_array = get_img_array(IMAGE_PATH,IMAGE_SIZE)
  image = get_image(IMAGE_PATH)
  model = get_model(MODEL_PATH)
  heatmap,pred_index = make_gradcam_heatmap(image_array,model,last_conv_layer_name,classifier_layer_names)
  heatmap = np.uint8(255 * heatmap) # rescaling the heatmap to a range 0-255
  jet = cm.get_cmap("jet")  # use jet colormap to colorize heatmap
  jet_colors = jet(np.arange(256))[:, :3]  # use rgb value of the colormap
  jet_heatmap = jet_colors[heatmap]
  jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
  jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
  jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
  superimposed_image = jet_heatmap * alpha + image # superimpose the heatmap on original image
  superimposed_image = keras.preprocessing.image.array_to_img(superimposed_image)
  print("Predicted Class: {}".format(class_label[pred_index]))
  plt.imshow(superimposed_image)

if __name__ == '__main__':
  display()


