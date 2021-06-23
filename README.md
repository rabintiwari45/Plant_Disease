# Plant Disease Classification and Detection

### THE ARTICLE FOR THE PROJECT CAN BE FOUND [Here](https://rabintiwari45.github.io/Portfolio/post/project-2/)

## Table of Content
* Demo
* Data Source
* Overview
* Installation
* Running the program
* References

## Demo
![image](https://github.com/rabintiwari45/Plant-Disease/blob/master/images/demo.png)
![image1](https://github.com/rabintiwari45/Plant-Disease/blob/master/images/demo1.png)
![image2](https://github.com/rabintiwari45/Plant-Disease/blob/master/images/demo2.png)

## Data Source

The dataset existing in current time are lab controlled which perform very poorly on real life condition with natural background, lightning, different stages of symptoms. The authors of Plant Doc Dataset have prepared their own dataset containing images from non controlled environment. So, we will use PlantDoc dataset for our project.

The PlantDoc dataset can be found [HERE](https://github.com/pratikkayal/PlantDoc-Dataset)

Further, we also used PlantVillage dataset (which is obtained from lab setting) to increase our model accuracy and to generate fake images using various GAN architecture.

The PlantVillage dataset can be found [HERE](https://github.com/spMohanty/PlantVillage-Dataset)

## Overview
This is an plant disease classification and detection web application. We used transfer learning to classify images. We also used Tensorflow Object Detection API
to train object detection models. Further, we also used various GAN architecture to generate fake images of plant leaves.



## Installation
The code is written in Python 3.7.10. The required packages and libraries for this project are:
```
pandas>=0.23.0
numpy>=1.19.1
Flask==0.10.1
gunicorn==20.1.0
itsdangerous==0.24
Jinja2==2.10
Werkzeug==0.14.1
MarkupSafe==0.23
tensorflow-cpu>=2.3.1
```
You can install all the library by running below command after cloning the repository.
```
pip install -r requirements.txt
```

## Running the program

After cloning the repository and adjusting the train and test path, you can run below command.

To run image classification model, change the working directory to folder Plant Classification and run the command.
```
python transfer learning on plantdoc.py
```
To run grad-cam, run the command.
```
python GRAD-CAM.py
```
To test new image, run the commmand.
```
python test.py
```
To run object detection model, you need to mannually run the files inside Plant Image Detection.

To test object detection model, change the working directory to Plant Image Detection and run the command.
```
python test.py
```
To run the GAN models, change the working directory to GANs.

To run DCGAN_WGAN_GP, run the command.
```
python DCGAN_with_GP.py
```
To run ProGAN, run the command.
```
python pro_gan.py
```
To run flask web app, run the command.
```
python app.py
```

## References
[PlantDoc: A Dataset for Visual Plant Disease Detection](https://arxiv.org/pdf/1911.10317.pdf)

[How to Train a Progressive Growing GAN in Keras](https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/)


### TO KNOW MORE ABOUT THE PROJECT YOU CAN CHECK THE ARTICLE [HERE](https://rabintiwari45.github.io/Portfolio/post/project-2/)








