# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive',force_remount=True)
root_dir = "/content/drive/shared drives/"

"""**google colab part:** Enabling TensorFlow GPU at Google Colab"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print( '\n\nThis error most likely means that this notebook is not ' 'configured to use a GPU. Change this in Notebook Settings via the ' 'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3)) 
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)

#We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
gpu()

"""**Enviroment setup**

Environment Constants
```
root_path: full path of directory where lfwa is exported. For example. If lfwa.zip is located on /root/ the root_path will be /root/lfwa/lfw2/

images_path: inner directory, where the hierarchy is /person folder/*.jpg. the value by default is root_path + “lfw2”.

TEST_CSV_PATH: full path where pairsDevTest.txt is located.

TRAIN_CSV_PATH: full path where pairsDevTrain.txt is located.

BATCH_NORM: enable/disable batch normalization 

IMG_HEIGHT, IMG_WIDTH: input shape, initial dimensions for the model’s input layer. Default is 105X105.

reg_lamda: the lambda value will be used for L2 regularization parameter. Default is 0.001
```
"""

#root_path = "/content/drive/My Drive/lfwa/lfw2/"
root_path= 'drive/Shared drives/DLEX2/lfw/lfwa/lfw2/'
images_path = root_path + "lfw2"
TEST_CSV_PATH = images_path +'/pairsDevTest.txt'
TRAIN_CSV_PATH = images_path + '/pairsDevTrain.txt'
PAIRS_CSV_PATH = "/content/drive/Shared drives/DLEX2/lfw/pairs.txt"
#TEST_CSV_PATH = root_path +'/pairsDevTest.txt'
#TRAIN_CSV_PATH = root_path + '/pairsDevTrain.txt'
IMG_HEIGHT = 105
IMG_WIDTH = 105
reg_lamda = 0.001
BATCH_NORM = False

"""Imports"""

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten,Lambda, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.callbacks import LearningRateScheduler
import time
import itertools
from scipy.misc import *
from matplotlib.pyplot import *
import matplotlib.pylab as plt
import tarfile, gzip
import cv2
import csv
import os
import numpy as np
import datetime
import pickle

"""**TensorBoard extension for notebook**"""

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard

"""**google colab part:**"""

!rm -rf ./logs/

"""***Data Reader, Loading and analysis***"""

images_classes = os.listdir(images_path)
with open(TRAIN_CSV_PATH, 'r') as csvfile:
    trainrows = list(csv.reader(csvfile, delimiter='\t'))[1:]
with open(TEST_CSV_PATH, 'r') as csvfile:
    testrows = list(csv.reader(csvfile, delimiter='\t'))[1:]
#Only for using pait.txt for test
#with open(PAIRS_CSV_PATH, 'r') as csvfile:
    #evalrows = list(csv.reader(csvfile, delimiter='\t'))[1:500]    
print("Finished loading Train and Test files.")

"""Some Data Analysis - Top examples, intersections etc"""

import glob
import pandas as pd

def plot_data():
  dataset = []
  for path in glob.iglob(os.path.join(images_path, "**", "*.jpg")):
    # print (path)
      person = path.split("/")[-2]
      dataset.append({"person":person, "num of images": path})
      
  dataset = pd.DataFrame(dataset)
  dataset = dataset.groupby("person").filter(lambda x: len(x) > 5)
  #dataset.head(10)
  #print (len(dataset.groupby("person").count()))
  dataset.groupby("person").count().plot(kind='hist', bins=200, figsize=(10,5))



def data_analysis():
  positems = []
  negitems = []
  print (len(testrows))
  for row in testrows :
      if (len(row)==3): #positive example
          name1 = row[0]
          positems.append(name1)
      if (len(row)==4): #negative example
          name1 = row[0]
          name2 = row[2]
          negitems.append(name1)
          negitems.append(name2)
  return positems, negitems

def intersections (negitems, positems):
  lst2 = [value for value in negitems if value in positems] 
  #print (lst2)
  lst3 = dict()
  for i in lst2:
    lst3[i] = lst3.get(i, 0) + 1
  print (lst3)
  print("intetsection len:",len(lst3))
  return lst3

def print_list():
  positems, negitems = data_analysis()
  lst3 = intersections(negitems, positems)
  for key in sorted(lst3, key=lst3.get, reverse=True):
    print(key, lst3[key])

def get_list(positems,negitems):
  counts = dict()
  counts_n = dict()
  for i in positems:
   counts[i] = counts.get(i, 0) + 1
  for i in negitems:
    counts_n[i] = counts_n.get(i, 0) + 1 
  return counts, counts_n


#for key in sorted(counts, key=counts.get, reverse=True):
 #   print(key, counts[key])  
positems,negitems = data_analysis()
counts, counts_n = get_list(positems,negitems)

for key in sorted(counts_n, key=counts_n.get, reverse=True):
    print(key, counts_n[key])

# Run this to get histogram of images quantities per person, first time would take a while
#plot_data()

"""images loader, decoder"""

#input processing
def load_image(images_path, name, number, os='lnx'):
    number = int(number)
    filename = "{0}/{1}/{1}_{2:04d}.jpg".format(images_path,name,number) # for linux os
    if (os=='win') : filename = "{0}\\{1}\\{1}_{2:04d}.jpg"
   # if (os == 'lnx') : 
    # <images_path>/<name>/<name>_<number 04d>.jpg   
    filename = filename.format(images_path, name, int(number))
    return cv2.imread(filename)

def decode_img(img):

    img = np.array(img / 255, dtype='float32')
    img2 = cv2.resize(img,(IMG_HEIGHT,IMG_WIDTH))
    return img2

"""**Building Datasets structure**

Build Datasets from positive and negative examples
"""

def load_pos_example(row):
    #same person, two images, of the format: name[0], image1[1], image[2]
    name = row[0]   
    imgnum1 = row[1]
    img1 = load_image(images_path, name, imgnum1)    
    imgnum2 = row[2]
    img2 = load_image(images_path, name, imgnum2)
    img1_dec = decode_img(img1)
    img2_dec = decode_img(img2)   
    return img1_dec, img2_dec

def load_neg_example(row):
    #different persons, two images, of the format: name[0], image1[1], name [2], image[3]
    name1 = row[0]
    imgnum1 = row[1]
    img1 = load_image(images_path, name1, imgnum1)
    
    name2 = row[2]
    imgnum2 = row[3]
    img2 = load_image(images_path, name2, imgnum2)

    img1_dec = decode_img(img1)
    img2_dec = decode_img(img2)

    return img1_dec, img2_dec

def make_dataset(trainrows):
    
    n = len(trainrows)
    num_of_examples = len(trainrows) //2
    print ("creating dataset",num_of_examples)
    first_imgs = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, 3))
    second_imgs = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, 3))
    targets = np.zeros((n, 1))
    targets.astype(float)
    i = 0

    for row in trainrows :
      
        if (i < n):
            if (len(row)==3): #positive example
                img1, img2 = load_pos_example(row)
                first_imgs[i] = img1
                second_imgs[i] = img2
                targets[i]=1.0
                i += 1
            if (len(row)==4): #negative example
                img1, img2 = load_neg_example(row)
                first_imgs[i] = img1
                second_imgs[i] = img2
                targets[i]=0.0
                i += 1
    print ("finish creating dataset")
    return first_imgs, second_imgs, targets

"""additional funcations we will use when fitting and compiling the model"""

def lr_scheduler(epoch, lr):
#'''Reducing learning rate implementation'''
    decay_rate = 0.99
    decay_step = 1
    if epoch:# % decay_step == 0 and epoch:
      if epoch < 30:
        return lr * decay_rate
    return lr

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
cbks = [tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),tensorboard_callback]

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

"""Plot function for graphs"""

import matplotlib.pyplot as plt
def PlotResults ( history, mode = 0 ):
  # Plot training & validation accuracy values
  if mode == 1:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'],color ='darkgreen')
  else:
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])

  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'],color ='darkgreen')
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

"""Building and setting up the network"""

def build_siamese_model():

    input_shape = (IMG_HEIGHT,IMG_WIDTH,3)
    w_init = initializers.RandomNormal(mean=0.0, stddev=0.01)
    b_init = initializers.RandomNormal(mean=0.5, stddev=0.01)
    batch_normalization = BATCH_NORM
    
    # Convolutional Neural Network
    cnn = Sequential()
    cnn.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,kernel_initializer=w_init, kernel_regularizer=l2(reg_lamda)))#(2e-4))) reg_lamda 
    if batch_normalization: cnn.add(BatchNormalization())
    cnn.add(Dropout(0.3))
    cnn.add(MaxPooling2D(strides=2))
   # cnn.add(Dropout(0.1))
    cnn.add(Conv2D(128, (7,7), activation='relu',kernel_initializer=w_init,bias_initializer=b_init, kernel_regularizer=l2(reg_lamda)))
    if batch_normalization: cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(strides=2))
  #  cnn.add(Dropout(0.1))
    cnn.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=w_init,bias_initializer=b_init, kernel_regularizer=l2(reg_lamda)))
    if batch_normalization: cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(strides=2))
    #cnn.add(Dropout(0.1))
    cnn.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=w_init,bias_initializer=b_init, kernel_regularizer=l2(reg_lamda)))
    if batch_normalization: cnn.add(BatchNormalization())
    cnn.add(Flatten())
    cnn.add(Dense(4096, activation='sigmoid',kernel_regularizer=l2(2e-4),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.2),bias_initializer=b_init))#1e-3
    cnn.add(Dropout(0.02))
    #Define Inputs
    image_X1 = tf.keras.Input(input_shape)
    image_X2 = tf.keras.Input(input_shape)
    
    #Recieving outputs by processing each image over the model
    output1 = cnn(image_X1)
    output2 = cnn(image_X2)
 
    #Define a costume Lambda layer for distance using euclidean distance calculation
    distance = tf.keras.layers.Lambda (lambda dist:(K.abs(dist[1] - dist[0])))

    #Create a distance layer, receiving 2 outputs and returns the distance
    distance_layer = distance([output1, output2])

    #connected final dense layer over distance layer
    preds = Dense(1, activation='sigmoid')(distance_layer)

    #build model
    siamese_model = models.Model(inputs=[image_X1, image_X2], outputs=preds)

    return siamese_model


def run_siamese_model (siamese_model, X1_All, X2_All, y_train, X1_test, X2_test, y_test ,lr = 0.01,opt = 1 ,loss = 1): 

    # Split the data to train and validation
    X1_train, X1_valid, X2_train, X2_valid, y_train, y_valid = train_test_split(X1_All, X2_All, y_train, test_size=0.20, shuffle= True)
   
    #Define different optimizers
    if opt == 1:
      print("[Training model..SGD")
      #opt_func = optimizers.SGD(lr=lr, momentum=0.7,nesterov=True)#, nesterov=True)
      opt_func = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.7,nesterov=True)#, nesterov=True) #decay=1e-6 amsgrad=False # lr= 0.02
   
    if opt == 2:
      print("[Training model..Adam")
      opt_func = optimizers.Adam(lr = lr, beta_1=0.9, beta_2=0.999, amsgrad=True) #lr = 0.01
   
    if opt == 3:
      print("[Training model..RMSprop")
      opt_func = optimizers.RMSprop(lr=lr, rho=0.9)

    if loss == 1:
      siamese_model.compile(loss=['binary_crossentropy'], metrics=['binary_accuracy'],optimizer=opt_func)#binary_accuracy binary_crossentropy contrastive_loss contrastive_loss
    else:  
      siamese_model.compile(loss=contrastive_loss,metrics=[accuracy],optimizer=opt_func)
   
    print("Training model..lr,loss,opt.", lr,loss ,opt )
  
    history = siamese_model.fit([X1_train, X2_train], y_train, batch_size=64, validation_data=([X1_valid, X2_valid], y_valid), callbacks=cbks, epochs=50,verbose=2)#, callbacks=cbks
    print("[Predicting images similarity...")
    predictions = siamese_model.predict([X1_test, X2_test])
    print('\n# Evaluate on test data')
    results = siamese_model.evaluate([X1_test, X2_test], y_test, batch_size=64,verbose=1)
    print('test loss, test acc:', results)
  

    return history,predictions

"""**Data Loading**

Next 2 cells are for loading datasets: **You should run ONLY one of them.** 
The first one loads datasets from local path, according to the directory path that was given on enviroments parameters: this should be done 1 time and takes a while.
the second cell is used ONLY if you work with our **shared drive** and have access premissions to pre-saved datasets, so there is no need to reload all the data from files. we used this option during development to save loading time.
"""

#Local loading of train and test datasets

X1_All, X2_All, y_train = make_dataset(trainrows)
X1_test, X2_test, y_test = make_dataset(testrows)

"""**google colab part:** Download datasets files from shared drive"""

#Remote downloading of saved datasets in shared drive

#np.savez('drive/Shared drives/DLEX2/Train.npz', X1_All=X1_All, X2_All=X2_All,y_train = y_train)
data = np.load('drive/Shared drives/DLEX2/Train.npz')
X1_All = data['X1_All']
X2_All = data['X2_All']
y_train = data['y_train']
#np.savez('drive/Shared drives/DLEX2/Test.npz', X1_test=X1_test, X2_test=X2_test,y_test = y_test)
data = np.load('drive/Shared drives/DLEX2/Test.npz')
#data = np.load('drive/Shared drives/DLEX2/Val.npz')
X1_test = data['X1_test']
X2_test = data['X2_test']
y_test = data['y_test']

print (" Upload file Train shape , Test shape  ",X1_All.shape,X2_test.shape)

"""**Main**

in this part we build the model object: sm model.
"""

sm = build_siamese_model()
sm.summary()

"""**Experiments**

in this part, we run the model with different parameters settings. we can call the run_siamese_model with different values for optimizer, loss function and learning rate.

**Note:** before each experiment we run (execution of run_siamese_model), we need to create new sm model, since the model in changing according to the values we set.

The process of experiment is:

1. *sm = build_siamese_model()*

2. *history,pred = run_siamese_model(sm, X1_All, X2_All, y_train, X1_test, X2_test, y_test,lr=0.005)*
"""

history,pred = run_siamese_model(sm, X1_All, X2_All, y_train, X1_test, X2_test, y_test,lr=0.005)
PlotResults (history)

"""Lets try with bigger learning rate"""

history,pred1 = run_siamese_model(sm, X1_All, X2_All, y_train, X1_test, X2_test, y_test,lr=0.02)
PlotResults (history)

"""Running with the contrastive_loss"""

history ,pred2 = run_siamese_model(sm, X1_All, X2_All, y_train, X1_test, X2_test, y_test,lr=0.02,loss = 0)
PlotResults (history,mode =1 )

"""Leta try with diff optimizer Adam"""

history ,pred3 =  run_siamese_model(sm, X1_All, X2_All, y_train, X1_test, X2_test, y_test,lr=0.02,opt = 2)
PlotResults (history)

"""Another round with Adam  with smaller learning rate"""

history ,pred4 =  run_siamese_model(sm, X1_All, X2_All, y_train, X1_test, X2_test, y_test,lr=0.0002,opt = 2)
PlotResults (history)

"""Look like we have 65 with Adam let try other Optimizer Lets try with RMSprop optimizer"""

history ,pred5 =  run_siamese_model(sm, X1_All, X2_All, y_train, X1_test, X2_test, y_test,lr=0.001,opt = 3)
PlotResults (history)

history ,pred6 =  run_siamese_model(sm, X1_All, X2_All, y_train, X1_test, X2_test, y_test,lr=0.01,opt = 3)
PlotResults (history)

history ,pred4 =  run_siamese_model(sm, X1_All, X2_All, y_train, X1_test, X2_test, y_test,lr=0.0001,opt = 3,loss = 0)
PlotResults (history,mode = 1)

"""Lets see where the are best model missed - on pictures of the same persons"""

predicted_classes = (pred4 >= 0.5).astype(int)
t= predicted_classes - y_test 
print (t[3],pred[3],predicted_classes[3],y_test[3])
print (t[2],pred[2],predicted_classes[2],y_test[2])
print (t[30:60])
f, ax = plt.subplots(2,2, figsize=(10,10))
ax[0][0].imshow(X1_test[6],cmap='gray')
ax[0][1].imshow(X2_test[6],cmap='gray')
ax[1][0].imshow(X1_test[2],cmap='gray')
ax[1][1].imshow(X2_test[2],cmap='gray')

"""Lets see where the model missed on pictures with different person"""

predicted_classes = (pred4 >= 0.5).astype(int)
t= predicted_classes - y_test 
print (t[3],pred4[3],predicted_classes[3],y_test[3])
print (t[2],pred4[2],predicted_classes[2],y_test[2])
print (t[4],pred4[4],predicted_classes[4],y_test[4])
print (t[500:520])
f, ax = plt.subplots(2,2, figsize=(10,10))
ax[0][0].imshow(X1_test[501],cmap='gray')
ax[0][1].imshow(X2_test[501],cmap='gray')
ax[1][0].imshow(X1_test[509],cmap='gray')
ax[1][1].imshow(X2_test[509],cmap='gray')

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs/fit

sm = build_siamese_model()
history,pred5 = run_siamese_model(sm, X1_All, X2_All, y_train, X1_test, X2_test, y_test,lr=0.02)
PlotResults (history)

sm = build_siamese_model()
history,pred6 = run_siamese_model(sm, X1_All, X2_All, y_train, X1_test, X2_test, y_test,lr=0.0002, opt = 2)
PlotResults (history)

sm = build_siamese_model()
history,pred6 = run_siamese_model(sm, X1_All, X2_All, y_train, X1_test, X2_test, y_test,lr=0.0002, opt = 3)
PlotResults (history)