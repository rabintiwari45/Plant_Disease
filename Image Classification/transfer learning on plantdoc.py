import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization,Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications import InceptionResNetV2

TRAINING_DATA_PATH = '/content/drive/MyDrive/plantdoc/PlantDoc-Dataset/train'
TESTING_DATA_PATH = '/content/drive/MyDrive/plantdoc/PlantDoc-Dataset/test'

IMAGE_SIZE = [224,224]
BATCH_SIZE = 32
LEARNING_RATE = 0.009
MOMENTUM = 0.9 
EPOCHS = 200
PATIENCE = 30
CLASSES = len(glob('data/train'))

def get_data():
  train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range = 0.2,
                                   zoom_range=0.2,
                                   horizontal_flip = True)
  test_datagen = ImageDataGenerator(rescale=1./255)
  train_set = train_datagen.flow_from_directory(TRAINING_DATA_PATH,
                                                 target_size = (224,224),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'categorical',
                                                 shuffle=True)

  test_set = test_datagen.flow_from_directory(TESTING_DATA_PATH,
                                            target_size = (224,224),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'categorical',
                                            shuffle = False)
  return train_set,test_set

def get_model(model=None):
  if model=='MobileNet':
    model = MobileNet(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)
  else:
    model = InceptionResNetV2(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)
  
  for layer in model.layers[:-20]:
    layer.trainable = False 

  layers = GlobalAveragePooling2D()(model.output)
  layers = Dense(1024, activation='relu',name='fc1')(layers)
  layers = Dropout(0.5,name='Dropout_1')(layers)
  layers = BatchNormalization()(layers)
  layers = Dense(512, activation='relu',name='fc2')(layers)
  layers = Dropout(0.5,name = 'Dropout')(layers)
  prediction = Dense(27,activation = 'softmax',name='output')(layers)

  model = Model(inputs=model.input,outputs= prediction)

  return model

def train(algo=None):
  model = get_model(algo)
  train,test = get_data()
  adam=optimizers.SGD(lr=LEARNING_RATE,momentum=MOMENTUM)
  monitor = EarlyStopping(monitor = 'val_accuracy',min_delta=1e-3,verbose=1,restore_best_weights=True,patience=30)
  model.compile(
        loss = 'categorical_crossentropy',
        optimizer = adam,
        metrics = ['accuracy']
  )
  cnn_model = model.fit_generator(train,
        validation_data = test,
        epochs = EPOCHS,
        steps_per_epoch = len(train)//10
        #callbacks = [monitor]

  )
  y_pred = model.predict(test)
  return cnn_model,y_pred

def plot_loss(model):
  loss_train = model.history['loss']
  loss_val = model.history['val_loss']
  epochs = range(1,EPOCHS+1)
  plt.plot(epochs,loss_train , 'g',label = 'Training Loss')
  plt.plot(epochs,loss_val,'b', label ='validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel("loss")
  plt.legend()
  plt.show()

def plot_accuracy(model):
  acc_train = model.history['accuracy']
  acc_val = model.history['val_accuracy']
  epochs = range(1,EPOCHS+1)
  plt.plot(epochs, acc_train , 'g',label = 'Training Accuracy')
  plt.plot(epochs,acc_val,'b', label ='validation Accuracy')
  plt.title('Training and validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel("accuracy")
  plt.legend()
  plt.show()

def con_matrix(model,y_pred):
  train,test = get_data()
  #y_pred = model.predict(test)
  y_pred = np.argmax(y_pred,axis=1)
  true_class = test.classes
  class_label = list(test.class_indices.keys())
  matrix = confusion_matrix(true_class,y_pred)
  plt.figure(figsize=(15,15))
  heatmap_confusion = sns.heatmap(matrix, annot=True, fmt="d");
  report = classification_report(true_class, y_pred, target_names=class_label,zero_division=1)
  return heatmap_confusion,report




if __name__ == '__main__':
  model,y_pred = train(algo='MobileNet')
  plot_loss(model)
  plot_accuracy(model)
  heatmap,report = con_matrix(model,y_pred)
  print(report)

