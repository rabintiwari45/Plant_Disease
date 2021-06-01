import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from IPython.display import Image, display
from tensorflow.keras.models import load_model


MODEL_PATH = '/content/drive/MyDrive/plantdoc/PlantDoc-Dataset/model_mobilenet_6525.h5'
IMAGE_PATH = '/content/drive/MyDrive/plantdoc/PlantDoc-Dataset/test/Apple Scab Leaf/052609%20Hartman%20Crabapple%20scab%20single%20leaf.JPG.jpg'
IMAGE_SIZE = (224,224)

class_label = ['Apple Scab Leaf','Apple leaf','Apple rust leaf','Bell_pepper leaf',
               'Bell_pepper leaf spot','Blueberry leaf','Cherry leaf','Corn Gray leaf spot',
               'Corn leaf blight','Corn rust leaf','Peach leaf','Potato leaf early blight',
               'Potato leaf late blight','Raspberry leaf','Soyabean leaf','Squash Powdery mildew leaf',
               'Strawberry leaf','Tomato Early blight leaf','Tomato Septoria leaf spot','Tomato leaf',
               'Tomato leaf bacterial spot','Tomato leaf late blight','Tomato leaf mosaic virus',
               'Tomato leaf yellow virus','Tomato mold leaf','grape leaf','grape leaf black rot'
              ]

def get_model(path):
  return load_model(path)

def get_image(img_path,img_size):
  img = keras.preprocessing.image.load_img(img_path,target_size=img_size)
  array = keras.preprocessing.image.img_to_array(img)
  array = np.expand_dims(array,axis=0)
  array = array/255.0
  return array

def test():
  model = get_model(MODEL_PATH)
  image = get_image(IMAGE_PATH,IMAGE_SIZE)
  pred = model.predict(image)
  pred_index = np.argmax(pred,axis=1)
  print("The predicted class of image is: {}".format(class_label[int(pred_index)]))
  display(Image(IMAGE_PATH))

if __name__ == '__main__':
  test()
