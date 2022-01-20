import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = "\\train2" 
IMG_SIZE = 250
MODEL_NAME = 'redConvolucional-gatos'
absPath = os.path.abspath(os.getcwd())

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def create_label(image_name):
  """ Create an one-hot encoded vector from image name"""
  word_label = image_name.split('.')[0]
  word_label= word_label.split(" ")[0]
  if word_label == 'm':
    return np.array([1,0])
  else:
    return np.array([0,1]) 

training_data = []
for img in tqdm(os.listdir(absPath+TRAIN_DIR)):
  path = os.path.join(absPath+TRAIN_DIR, img)
  img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
  img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
  training_data.append([np.array(img_data), create_label(img)])
  faces = face_cascade.detectMultiScale(img_data, 1.35, 1)
  for (x,y,w,h) in faces:

    roi_gray = img_data[y:y+h, x:x+w]
    roi_color = img_data[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes: #bucle que busca entre todos los ojos detectados en la imagen
      if (ey<h/2):
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(40,55,200),2)
        height,width = img_data.shape[0:2]
        startRow = int(height*.23)
        startCol = int(width*.15)
        endRow = int(height*.73)
        endCol = int(width*.85)
        img_data = img_data[startRow:endRow, startCol:endCol]


  cv2.imshow('img',img_data)
  #cv2.imshow('cropped',croppedImage)
  #cv2.imwrite('img.jpg',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()