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

TRAIN_DIR = "\\train" 
TEST_DIR = "\\test"
IMG_SIZE = 50
LR = 0.001
MODEL_NAME = 'model_svIAl'
absPath = os.path.abspath(os.getcwd())

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def set_label(image_name):
  word_label = image_name.split('.')[0]
  word_label= word_label.split(" ")[0]
  if word_label == 'm':
    return np.array([1,0])
  else:
    return np.array([0,1]) 

def load_data():
  training_data = []
  for img in tqdm(os.listdir(absPath+TRAIN_DIR)):
    path = os.path.join(absPath+TRAIN_DIR, img)
    img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
    faces = face_cascade.detectMultiScale(img_data, 1.35, 1)
    for (x,y,w,h) in faces:
      roi_gray = img_data[y:y+h, x:x+w]
      roi_color = img_data[y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale(roi_gray)
      i=0
      for (ex,ey,ew,eh) in eyes: #bucle que busca entre todos los ojos detectados en la imagen
        font = cv2.FONT_HERSHEY_SIMPLEX
        if (ey<h/2):
          height,width = img_data.shape[0:2]
          startRow = int(height*.23)
          startCol = int(width*.15)
          endRow = int(height*.73)
          endCol = int(width*.85)
          img_data = img_data[startRow:endRow, startCol:endCol]
      training_data.append([np.array(img_data), set_label(img)])
  shuffle(training_data)
  np.save('train_data.npy', training_data)
  return training_data

def create_test():
  testing_data = []
  for img in tqdm(os.listdir(absPath+TEST_DIR)):
   path = os.path.join(absPath+TEST_DIR,img)
   img_num = img.split('.')[0]
   img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
   img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
   testing_data.append([np.array(img_data), img_num])

  shuffle(testing_data)
  np.save('test_data.npy', testing_data)
  return testing_data

train_data = load_data()
test_data = create_test()

train = train_data[:-362]
test = train_data[-362:]
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]

tf.compat.v1.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 256, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 512, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 256, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
model.fit({'input': X_train}, {'targets': y_train}, n_epoch=30,
          validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=362, show_metric=True, run_id=MODEL_NAME)

d = test_data[0]
img_data, img_num = d

data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
prediction = model.predict([data])[0]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.imshow(img_data, cmap="gray")
print(f"mirando: {prediction[0]}, noMirando: {prediction[1]}")

fig=plt.figure(figsize=(16, 12))

for num, data in enumerate(test_data[:16]):

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(4, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label='NO mirando'
    else:
        str_label='Mirando'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()