import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from os import listdir
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model


generate_res = 3
generate_square = 2*32
channels = 3
preview_rows = 8
preview_column = 8
preview_margin = 16
noise_dim = 100
epochs = 3000
batch_size = 16
buffer_size = 175
image_shape = (64,64,3)
n_critic = 5
penalty_lambda = 10

PATH_TO_DATASET = '/content/drive/MyDrive/plantdoc/PlantDoc-Dataset/train'
DATA_PATH = '/content/gan'


def get_data(path):
  image_list = []
  folder_dir = listdir(path)
  for i in range(0,len(folder_dir)):
    image_dir = listdir(path + "/" + folder_dir[i])
    print(folder_dir[i])
    for j in range(0,len(image_dir)):
      image_path = path + "/" + folder_dir[i] + "/" + image_dir[j]
      image = cv2.imread(image_path)
      image = cv2.resize(image,(64,64))
      image_list.append(image)
  train_image_array = np.array(image_list)
  train_image_array = train_image_array.astype('float32')
  train_image_array = (train_image_array-127.5)/127.5
  train_dataset = tf.data.Dataset.from_tensor_slices(train_image_array).shuffle(buffer_size).batch(batch_size)
  return train_dataset 



def build_generator(noise_dim,channels):
  model = Sequential()

  model.add(Dense(4*4*256,activation = 'relu',input_dim = noise_dim))
  model.add(Reshape((4,4,256)))

  model.add(UpSampling2D())
  model.add(Conv2D(256,kernel_size = 3,padding='same'))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Activation('relu'))

  model.add(UpSampling2D())
  model.add(Conv2D(256,kernel_size=3,padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Activation("relu"))

  model.add(UpSampling2D())
  model.add(Conv2D(128,kernel_size=3,padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Activation("relu"))

  if generate_res>1:
    #model.add(UpSampling2D(size=(generate_res,generate_res)))
    model.add(UpSampling2D())
    model.add(Conv2D(128,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

  model.add(Conv2D(channels,kernel_size=3,padding="same"))
  model.add(Activation("tanh"))

  return model

def build_critic(image_shape):
  model = Sequential()

  model.add(Conv2D(32, kernel_size=3, input_shape=image_shape,padding="same"))
  model.add(LeakyReLU(alpha=0.2))

  model.add(Dropout(0.25))
  model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
  model.add(ZeroPadding2D(padding=((0,1),(0,1))))
  #model.add(BatchNormalization(momentum=0.8))
  model.add(LeakyReLU(alpha=0.2))

  model.add(Dropout(0.25))
  model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
  #model.add(BatchNormalization(momentum=0.8))
  model.add(LeakyReLU(alpha=0.2))

  model.add(Dropout(0.25))
  model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
  #model.add(BatchNormalization(momentum=0.8))
  model.add(LeakyReLU(alpha=0.2))

  model.add(Dropout(0.25))
  model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
  #model.add(BatchNormalization(momentum=0.8))
  model.add(LeakyReLU(alpha=0.2))

  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(1))

  return model

def critic_loss_wgangp(real_out,fake_out):
  real_loss = tf.reduce_mean(real_out)
  fake_loss = tf.reduce_mean(fake_out)
  return fake_loss - real_loss

def generator_loss_wgangp(fake_out):
  return -tf.reduce_mean(fake_out)

def GP(critic,batch_size,real_image,fake_image):
  epsilon = tf.random.normal([batch_size,1,1,1],0.0,1.0)
  diff = fake_image - real_image
  averaged_image = real_image + epsilon*diff
  with tf.GradientTape() as tape:
    tape.watch(averaged_image)
    averaged_out = critic(averaged_image,training=True)
  grads = tape.gradient(averaged_out,[averaged_image])[0]
  norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
  gp = tf.reduce_mean((norm - 1.0) ** 2)
  return gp

def train_step(real_image,generator,critic,batch_size,n_critic,noise_dim,penalty_lambda):
  critic_optimizer = Adam(learning_rate=1e-5,beta_1=0.5,beta_2=0.9)
  generator_optimizer = Adam(learning_rate=1e-5,beta_1=0.5,beta_2=0.9)
  if isinstance(real_image,tuple):
    real_image = real_image
  for i in range(n_critic):
    noise = tf.random.normal(shape=(batch_size,noise_dim))
    with tf.GradientTape() as tape:
      fake_image = generator(noise,training=True)
      fake_out = critic(fake_image,training=True)
      real_out = critic(real_image,training=True)
      critic_cost = critic_loss_wgangp(real_out,fake_out)
      gp = GP(critic,batch_size,real_image,fake_image)
      critic_loss = critic_cost + gp * penalty_lambda
    critic_gradient = tape.gradient(critic_loss,critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_gradient,critic.trainable_variables))
  noise = tf.random.normal(shape=(batch_size,noise_dim))
  with tf.GradientTape() as tape:
    generated_image = generator(noise,training=True)
    generated_out = critic(generated_image,training=True)
    generator_loss = generator_loss_wgangp(generated_out)
  generator_gradient = tape.gradient(generator_loss,generator.trainable_variables)
  generator_optimizer.apply_gradients(zip(generator_gradient,generator.trainable_variables))
  #return {"critic_loss":critic_loss,"generator_loss":generator_loss}
  return generator_loss,critic_loss

def save_images(cnt,noise,generator):
  image_array = np.full(( 
      preview_margin + (preview_rows * (generate_square+preview_margin)), 
      preview_margin + (preview_column * (generate_square+preview_margin)), 3), 
      255, dtype=np.uint8)
  
  generated_images = generator.predict(noise)

  generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  for row in range(preview_rows):
      for col in range(preview_column):
        r = row * (generate_square+16) + preview_margin
        c = col * (generate_square+16) + preview_margin
        image_array[r:r+generate_square,c:c+generate_square] \
            = generated_images[image_count] * 255
        image_count += 1

          
  output_path = os.path.join(DATA_PATH,'output')
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  filename = os.path.join(output_path,f"generated_image-{cnt}.jpg")
  im = Image.fromarray(image_array)
  im.save(filename)

def train(dataset, epochs):
  generator = build_generator(noise_dim,channels)
  critic = build_critic(image_shape=(64,64,3))
  fixed_seed = np.random.normal(0, 1, (preview_rows* preview_column,  noise_dim))
  start = time.time()

  for epoch in range(epochs):
    epoch_start = time.time()

    gen_loss_list = []
    critic_loss_list = []

    for image_batch in dataset:
      if image_batch.shape[0] == 4:
        break
      t = train_step(image_batch,generator,critic,batch_size,n_critic,noise_dim,penalty_lambda)
      gen_loss_list.append(t[0])
      critic_loss_list.append(t[1])

    g_loss = sum(gen_loss_list) / len(gen_loss_list)
    c_loss = sum(critic_loss_list) / len(critic_loss_list)

    epoch_elapsed = time.time()-epoch_start
    print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={c_loss},'\
           ' {hms_string(epoch_elapsed)}')
    
    
    if epoch % 100 == 0:
      save_images(epoch,fixed_seed,generator)

    elapsed = time.time()-start

if __name__ == '__main__':
  data = get_data(PATH_TO_DATASET)
  train(data,epochs)
